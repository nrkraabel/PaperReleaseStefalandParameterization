from typing import Optional

import torch
import torch.nn as nn
from models.neural_networks.transformer.advanced_encoders import TFTPositionalEncoding
from models.neural_networks.transformer.features_embedding import (
    PatchDepatcher,
    PatchTokenizer,
    StaticDecEmbeddingLSTM,
    StaticEncEmbedding,
)
from models.neural_networks.transformer.mask import MaskGenerator
from models.neural_networks.transformer.transformer_layers import TransformerBackbone


class Model(nn.Module):
    """
    MFFormer with PatchTST-style tokenization + TFT-style temporal conditioning.
    - Time series -> Patch tokens (per channel)
    - Append STATIC context token
    - Apply TFTPositionalEncoding on token-wise temporal features
    - TransformerBackbone encodes tokens
    - Shared probabilistic de-patcher reconstructs mu, sigma over [B, T, F]
    - Static head remains your StaticEncEmbedding -> StaticDec path (for reconstruction of statics)

    Expected batch_data_dict keys:
      'batch_x' : [B, T, F]  (time series)
      'batch_c' : [B, C]     (static)
      'batch_time_series_mask_index' : [B, T, F]  bool
      'batch_static_mask_index'      : [B, C]     bool
      'temporal_features'            : [B, T, K]
      'mode'                         : (passed to static decoder)
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        d_model = configs.d_model
        d_ffd = configs.d_ffd
        self.embed_dim = d_model

        self.time_series_variables = configs.time_series_variables
        self.static_variables = configs.static_variables
        self.static_variables_category = configs.static_variables_category
        self.static_variables_category_num = [
            len(configs.static_variables_category_dict[x]['class_to_index'])
            for x in configs.static_variables_category
        ]
        self.static_variables_numeric = [
            v for v in self.static_variables if v not in self.static_variables_category
        ]
        self.num_channels = len(self.time_series_variables)

        # Patch tokenization
        self.use_patches = configs.use_patches
        self.patch_len = configs.patch_len
        self.patch_stride = configs.patch_stride

        self.tokenizer = PatchTokenizer(
            d_model=d_model,
            patch_len=self.patch_len,
            stride=self.patch_stride,
            num_channels=self.num_channels,
            dropout=configs.dropout,
        )

        # Static embedding
        self.static_embedding = StaticEncEmbedding(
            self.static_variables_numeric,
            d_model,
            categorical_features=self.static_variables_category,
            categorical_features_num=self.static_variables_category_num,
            dropout=configs.dropout,
        )

        # TFT-style temporal conditioning on tokens
        self.positional_encoding = TFTPositionalEncoding(
            d_model, dropout=configs.dropout
        )

        # masking
        self.mask_generator = MaskGenerator(
            configs.mask_ratio_time_series,
            mask_ratio_static=configs.mask_ratio_static,
            min_window_size=configs.min_window_size,
            max_window_size=configs.max_window_size,
        )

        # Transformer backbone
        self.encoder = TransformerBackbone(
            d_model, configs.num_enc_layers, d_ffd, configs.num_heads, configs.dropout
        )

        self.encoder_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(configs.dropout)

        # shared probabilistic depatcher head
        self.depatcher = PatchDepatcher(
            d_model=d_model,
            patch_len=self.patch_len,
            stride=self.patch_stride,
            num_channels=self.num_channels,
        )

        # Static decoder head
        self.static_projection = StaticDecEmbeddingLSTM(
            self.static_variables_numeric,
            d_model,
            categorical_features=self.static_variables_category,
            categorical_features_num=self.static_variables_category_num,
            dropout=configs.dropout,
            add_input_noise=configs.add_input_noise,
        )

        # Small projection to align any residual dims if needed
        self.enc_2_dec_embedding = nn.Linear(d_model, d_model, bias=True)

        self.init_weights()

    def init_weights(self):
        def init_layer(layer):
            if hasattr(layer, "weight") and layer.weight is not None:
                nn.init.uniform_(
                    layer.weight, -self.configs.init_weight, self.configs.init_weight
                )
            if hasattr(layer, "bias") and layer.bias is not None:
                nn.init.uniform_(
                    layer.bias, -self.configs.init_bias, self.configs.init_bias
                )

        init_layer(self.tokenizer.patch_proj)
        init_layer(self.depatcher.head)
        init_layer(self.enc_2_dec_embedding)

        for emb in self.static_embedding.numerical_embeddings1.values():
            init_layer(emb)
        for emb in self.static_embedding.numerical_embeddings2.values():
            init_layer(emb)
        nn.init.uniform_(
            self.static_embedding.masked_values,
            -self.configs.init_weight,
            self.configs.init_weight,
        )

    def _token_temporal_features(
        self, temporal_features: Optional[torch.Tensor], T: int, Np: int
    ) -> Optional[torch.Tensor]:
        """
        Average per-patch temporal features and tile across channels to align with tokens.
        temporal_features: [B, T, K] or None
        returns: [B, F*Np, K] or None
        """
        if temporal_features is None:
            return None
        B, T_in, K = temporal_features.shape
        assert T_in == T, "temporal_features length must match time axis of batch_x"

        # Unfold and average over patch window: [B, T, K] -> [B, Np, K, P] -> [B, Np, K]
        tf = temporal_features.unfold(
            dimension=1, size=self.patch_len, step=self.patch_stride
        ).mean(dim=-1)
        # Repeat across channels to match token layout
        tf = tf.unsqueeze(1).repeat(1, self.num_channels, 1, 1)  # [B, F, Np, K]
        tf = tf.view(B, self.num_channels * tf.size(2), K)  # [B, F*Np, K]
        return tf

    def forward(self, batch_data_dict, is_mask: bool = True):
        batch_x = batch_data_dict['batch_x']  # [B, T, F]
        batch_c = batch_data_dict['batch_c']  # [B, C]
        m_ts = batch_data_dict['batch_time_series_mask_index']  # [B, T, F] bool
        m_st = batch_data_dict['batch_static_mask_index']  # [B, C] bool
        temporal_features = batch_data_dict.get(
            'temporal_features', None
        )  # [B, T, K] or None
        mode = batch_data_dict.get('mode', None)

        device = batch_x.device
        B, T, F = batch_x.shape
        assert F == self.num_channels, (
            "Input channels mismatch with config time_series_variables"
        )

        # Masking
        if is_mask:
            if m_ts.numel() == 0:
                _, m_ts = self.mask_generator(batch_x.shape, method="consecutive")
                _, m_st = self.mask_generator(batch_c.shape, method="isolated_point")
            m_ts = m_ts.to(device)
            m_st = m_st.to(device)

            m_missing_ts = torch.isnan(batch_x)
            m_missing_st = torch.isnan(batch_c)
            batch_x = batch_x.masked_fill(m_missing_ts, 0.0)
            batch_c = batch_c.masked_fill(m_missing_st, 0.0)

            m_ts = m_ts | m_missing_ts
            m_st = m_st | m_missing_st
        else:
            m_missing_ts = torch.isnan(batch_x)
            m_missing_st = torch.isnan(batch_c)

        #  Patch tokens
        tokens, Np = self.tokenizer(batch_x)  # [B, F*Np, d_model]

        # Token temporal features for TFT-style conditioning
        token_tf = self._token_temporal_features(
            temporal_features, T, Np
        )  # [B, F*Np, K] or None

        # Static tokens (grouped if available)
        # Prefer attribute set on self; fallback to configs to avoid __init__ changes.
        group_mask_dict = getattr(
            self, "group_mask_dict", getattr(self.configs, "group_mask_dict", None)
        )
        if group_mask_dict and len(group_mask_dict) > 0:
            # map static var name -> column index in batch_c
            static_name2idx = {name: i for i, name in enumerate(self.static_variables)}
            Bc, C = batch_c.shape
            assert Bc == B, "batch_c batch size must match batch_x"

            static_token_list = []
            for _, g_vars in group_mask_dict.items():
                # mask everything EXCEPT this group's variables; also respect existing static mask m_st
                m_group = torch.ones(B, C, dtype=torch.bool, device=device)
                keep_idx = [static_name2idx[v] for v in g_vars if v in static_name2idx]
                if len(keep_idx) > 0:
                    m_group[:, keep_idx] = False

                enc_g = self.static_embedding(
                    batch_c,
                    feature_order=self.static_variables,
                    masked_index=(m_st | m_group),
                )  # [B, d_model]
                static_token_list.append(enc_g.unsqueeze(1))  # [B, 1, d_model]

            if len(static_token_list) == 0:
                # safety fallback: single token over all statics
                enc_c = self.static_embedding(
                    batch_c, feature_order=self.static_variables, masked_index=m_st
                )  # [B, d_model]
                static_tokens = enc_c.unsqueeze(1)  # [B, 1, d_model]
            else:
                static_tokens = torch.cat(static_token_list, dim=1)  # [B, G, d_model]
        else:
            # original single-token behavior
            enc_c = self.static_embedding(
                batch_c, feature_order=self.static_variables, masked_index=m_st
            )  # [B, d_model]
            static_tokens = enc_c.unsqueeze(1)  # [B, 1, d_model]

        # Concat dynamic tokens + static tokens
        enc_x = torch.cat(
            [tokens, static_tokens], dim=1
        )  # [B, F*Np + (G or 1), d_model]

        # Compose TFT temporal-features for all tokens
        if token_tf is not None:
            num_static_tokens = static_tokens.size(1)
            zero_static_tf = torch.zeros(
                token_tf.size(0),
                num_static_tokens,
                token_tf.size(2),
                device=token_tf.device,
            )
            concat_tf = torch.cat(
                [token_tf, zero_static_tf], dim=1
            )  # [B, F*Np + (G or 1), K]
        else:
            concat_tf = None

        # Encode
        enc_x = self.positional_encoding(
            enc_x, temporal_features=concat_tf
        )  # [B, L, d_model]
        hidden = self.encoder(enc_x)  # [B, L, d_model]
        hidden = self.encoder_norm(hidden)
        hidden = self.dropout(hidden)
        hidden = self.enc_2_dec_embedding(hidden)

        # Split token/static
        num_static_tokens = static_tokens.size(1)
        token_states = hidden[:, :-num_static_tokens, :]  # [B, F*Np, d_model]
        static_states = hidden[:, -num_static_tokens:, :]  # [B, (G or 1), d_model]

        # Reconstruct μ, σ over [B, T, F]
        mu, sigma = self.depatcher(token_states, T)  # [B, T, F] each

        # Static decoding
        # Pool grouped static tokens
        static_pooled = static_states.mean(dim=1)  # [B, d_model]
        outputs_static, s_start, s_end = self.static_projection(
            static_pooled, feature_order=self.static_variables, mode=mode
        )

        # Raw encoder latents (pre-depatcher-head), for downstream consumers that
        # want the learned d_model representation rather than the reconstructed
        # [B, T, F] values used for the pretraining reconstruction loss.
        encoder_hidden_time_series = self._patch_states_to_time(
            token_states, T
        )  # [B, T, d_model]
        encoder_hidden_static = static_pooled  # [B, d_model]

        output_dict = {
            # legacy names system
            'outputs_time_series': mu,
            'outputs_static': outputs_static,
            # new probabilistic fields for NLL/quantiles later
            'outputs_time_series_mu': mu,
            'outputs_time_series_sigma': sigma,
            # raw encoder embeddings (d_model-wide, not decoded to variable space)
            'encoder_hidden_time_series': encoder_hidden_time_series,
            'encoder_hidden_static': encoder_hidden_static,
            # masks
            'masked_time_series_index': m_ts,
            'masked_static_index': m_st,
            'masked_missing_time_series_index': m_missing_ts,
            'masked_missing_static_index': m_missing_st,
            # static decoder bookkeeping
            'static_variables_dec_index_start': s_start,
            'static_variables_dec_index_end': s_end,
        }
        return output_dict

    def _patch_states_to_time(self, token_states: torch.Tensor, T: int) -> torch.Tensor:
        """Overlap-add per-patch d_model embeddings (pooled across channels) back
        to per-timestep resolution, mirroring PatchDepatcher's reconstruction
        but keeping the full latent vector instead of collapsing it to scalars.

        token_states: [B, F*Np, D] channel-major -> returns [B, T, D]
        """
        B, FNp, D = token_states.shape
        F = self.num_channels
        Np = FNp // F
        h = token_states.view(B, F, Np, D).mean(dim=1)  # [B, Np, D]

        out = h.new_zeros(B, T, D)
        counts = h.new_zeros(B, T, 1)
        for p in range(Np):
            t0 = p * self.patch_stride
            if t0 >= T:
                break
            t1 = min(t0 + self.patch_len, T)
            out[:, t0:t1, :] += h[:, p, :].unsqueeze(1)
            counts[:, t0:t1, :] += 1.0
        counts = counts.clamp_min(1.0)
        return out / counts

    # old verison
    # def forward(self, batch_data_dict, is_mask: bool = True):
    #     batch_x = batch_data_dict['batch_x']    # [B, T, F]
    #     batch_c = batch_data_dict['batch_c']    # [B, C]
    #     m_ts = batch_data_dict['batch_time_series_mask_index']  # [B, T, F] bool
    #     m_st = batch_data_dict['batch_static_mask_index']       # [B, C] bool
    #     temporal_features = batch_data_dict.get('temporal_features', None)  # [B, T, K] or None
    #     mode = batch_data_dict.get('mode', None)

    #     device = batch_x.device
    #     B, T, F = batch_x.shape
    #     assert F == self.num_channels, "Input channels mismatch with config time_series_variables"

    #     #Masking
    #     if is_mask:
    #         if m_ts.numel() == 0:
    #             _, m_ts = self.mask_generator(batch_x.shape, method="consecutive")
    #             _, m_st = self.mask_generator(batch_c.shape, method="isolated_point")
    #         m_ts = m_ts.to(device)
    #         m_st = m_st.to(device)

    #         m_missing_ts = torch.isnan(batch_x)
    #         m_missing_st = torch.isnan(batch_c)
    #         batch_x = batch_x.masked_fill(m_missing_ts, 0.0)
    #         batch_c = batch_c.masked_fill(m_missing_st, 0.0)

    #         m_ts = m_ts | m_missing_ts
    #         m_st = m_st | m_missing_st
    #     else:
    #         m_missing_ts = torch.isnan(batch_x)
    #         m_missing_st = torch.isnan(batch_c)

    #     #Patch tokens
    #     tokens, Np = self.tokenizer(batch_x)  # [B, F*Np, d_model]

    #     #Build token temporal features for TFT (plus a dummy for the static token)
    #     token_tf = self._token_temporal_features(temporal_features, T, Np)  # [B, F*Np, K] or None

    #     #Static token (context)
    #     enc_c = self.static_embedding(batch_c, feature_order=self.static_variables, masked_index=m_st)  # [B, d_model]
    #     static_token = enc_c.unsqueeze(1)  # [B, 1, d_model]

    #     # concat tokens + static
    #     enc_x = torch.cat([tokens, static_token], dim=1)  # [B, F*Np+1, d_model]

    #     #Compose TFT temporal-features for all tokens
    #     if token_tf is not None:
    #         zero_static_tf = torch.zeros(token_tf.size(0), 1, token_tf.size(2), device=token_tf.device)
    #         concat_tf = torch.cat([token_tf, zero_static_tf], dim=1)  # [B, F*Np+1, K]
    #     else:
    #         concat_tf = None

    #     # TFT-style temporal conditioning, then encode
    #     enc_x = self.positional_encoding(enc_x, temporal_features=concat_tf)   # [B, L, d_model]
    #     hidden = self.encoder(enc_x)                                           # [B, L, d_model]
    #     hidden = self.encoder_norm(hidden)
    #     hidden = self.dropout(hidden)
    #     hidden = self.enc_2_dec_embedding(hidden)

    #     #Split token/static
    #     token_states = hidden[:, :-1, :]  # [B, F*Np, d_model]
    #     static_state = hidden[:, -1, :]   # [B, d_model]

    #     #Reconstruct μ, σ over [B, T, F]
    #     mu, sigma = self.depatcher(token_states, T)  # [B, T, F] each

    #     # Static decoding
    #     outputs_static, s_start, s_end = self.static_projection(
    #         static_state, feature_order=self.static_variables, mode=mode
    #     )
    #     #eventually this will be simplfied
    #     output_dict = {
    #         # legacy names system
    #         'outputs_time_series': mu,
    #         'outputs_static': outputs_static,

    #         # new probabilistic fields for NLL/quantiles later
    #         'outputs_time_series_mu': mu,
    #         'outputs_time_series_sigma': sigma,

    #         # masks
    #         'masked_time_series_index': m_ts,
    #         'masked_static_index': m_st,
    #         'masked_missing_time_series_index': m_missing_ts,
    #         'masked_missing_static_index': m_missing_st,

    #         # static decoder bookkeeping
    #         'static_variables_dec_index_start': s_start,
    #         'static_variables_dec_index_end': s_end,
    #     }
    #     return output_dict
