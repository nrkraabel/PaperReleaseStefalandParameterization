import torch
from models.neural_networks.transformer.advanced_encoders import TFTPositionalEncoding
from models.neural_networks.transformer.features_embedding import (
    StaticDecEmbeddingLSTM,
    StaticEncEmbedding,
    TimeSeriesDecEmbeddingLSTM,
    TimeSeriesEncEmbedding,
)
from models.neural_networks.transformer.mask import MaskGenerator
from models.neural_networks.transformer.transformer_layers import TransformerBackbone
from torch import nn


class Model(nn.Module):
    """
    MFFormer with Temporal Fusion Transformer (TFT) style temporal encoding.

    Uses variable selection network for temporal features, gated linear units,
    and multi-head temporal attention for pattern capture.
    """

    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        embed_dim = configs.d_model
        self.embed_dim = embed_dim
        d_ffd = configs.d_ffd

        self.time_series_variables = configs.time_series_variables
        self.static_variables = configs.static_variables
        self.static_variables_category = configs.static_variables_category
        self.static_variables_category_num = [
            len(configs.static_variables_category_dict[x]['class_to_index'])
            for x in configs.static_variables_category
        ]
        self.static_variables_numeric = [
            var
            for var in self.static_variables
            if var not in self.static_variables_category
        ]

        # Norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # Feature embeddings (unchanged from original)
        self.time_series_embedding = TimeSeriesEncEmbedding(
            configs.time_series_variables, embed_dim, dropout=configs.dropout
        )
        self.static_embedding = StaticEncEmbedding(
            self.static_variables_numeric,
            embed_dim,
            categorical_features=self.static_variables_category,
            categorical_features_num=self.static_variables_category_num,
            dropout=configs.dropout,
        )

        # TFT-style temporal encoding
        self.positional_encoding = TFTPositionalEncoding(
            embed_dim, dropout=configs.dropout
        )

        # Mask generator (unchanged)
        self.mask_generator = MaskGenerator(
            configs.mask_ratio_time_series,
            mask_ratio_static=configs.mask_ratio_static,
            min_window_size=configs.min_window_size,
            max_window_size=configs.max_window_size,
        )

        # Transformer backbone (unchanged)
        self.encoder = TransformerBackbone(
            embed_dim, configs.num_enc_layers, d_ffd, configs.num_heads, configs.dropout
        )

        # Decoder layers (unchanged)
        self.enc_2_dec_embedding = nn.Linear(embed_dim, embed_dim, bias=True)

        # Projection layers (unchanged)
        self.time_series_projection = TimeSeriesDecEmbeddingLSTM(
            configs.time_series_variables,
            embed_dim,
            dropout=configs.dropout,
            add_input_noise=configs.add_input_noise,
        )
        self.static_projection = StaticDecEmbeddingLSTM(
            self.static_variables_numeric,
            embed_dim,
            categorical_features=self.static_variables_category,
            categorical_features_num=self.static_variables_category_num,
            dropout=configs.dropout,
            add_input_noise=configs.add_input_noise,
        )
        self.dropout = nn.Dropout(configs.dropout)
        self.init_weights()

    def init_weights(self):
        def init_layer(layer):
            nn.init.uniform_(
                layer.weight, -self.configs.init_weight, self.configs.init_weight
            )
            nn.init.uniform_(
                layer.bias, -self.configs.init_bias, self.configs.init_bias
            )

        layers_to_init = [
            *self.time_series_embedding.embeddings1.values(),
            *self.time_series_embedding.embeddings2.values(),
            *self.static_embedding.numerical_embeddings1.values(),
            *self.static_embedding.numerical_embeddings2.values(),
        ]

        for layer in layers_to_init:
            init_layer(layer)

        nn.init.uniform_(
            self.time_series_embedding.masked_values,
            -self.configs.init_weight,
            self.configs.init_weight,
        )
        nn.init.uniform_(
            self.static_embedding.masked_values,
            -self.configs.init_weight,
            self.configs.init_weight,
        )

    def forward(self, batch_data_dict, is_mask=True):
        batch_x = batch_data_dict['batch_x']
        batch_c = batch_data_dict['batch_c']
        masked_time_series_index = batch_data_dict['batch_time_series_mask_index']
        masked_static_index = batch_data_dict['batch_static_mask_index']

        # Extract temporal features for TFT variable selection
        temporal_features = batch_data_dict.get(
            'temporal_features', None
        )  # [batch_size, seq_len, 7]
        print(temporal_features)

        # Generate mask (unchanged from original)
        if is_mask:
            if masked_time_series_index.numel() == 0:
                _, masked_time_series_index = self.mask_generator(
                    batch_x.shape, method="consecutive"
                )
                _, masked_static_index = self.mask_generator(
                    batch_c.shape, method="isolated_point"
                )

            masked_time_series_index = masked_time_series_index.to(batch_x.device)
            masked_static_index = masked_static_index.to(batch_x.device)

            masked_missing_time_series_index = torch.isnan(batch_x)
            masked_missing_static_index = torch.isnan(batch_c)
            batch_x = batch_x.masked_fill(masked_missing_time_series_index, 0)
            batch_c = batch_c.masked_fill(masked_missing_static_index, 0)

            masked_time_series_index = (
                masked_time_series_index | masked_missing_time_series_index
            )
            masked_static_index = masked_static_index | masked_missing_static_index
        else:
            masked_missing_time_series_index = torch.isnan(batch_x)
            masked_missing_static_index = torch.isnan(batch_c)

        # Feature embeddings (unchanged)
        enc_x = self.time_series_embedding(
            batch_x,
            feature_order=self.time_series_variables,
            masked_index=masked_time_series_index,
        )
        enc_c = self.static_embedding(
            batch_c,
            feature_order=self.static_variables,
            masked_index=masked_static_index,
        )

        # Concat time series and static features
        enc_x = torch.cat([enc_x, enc_c[:, None, :]], dim=1)

        # Apply TFT-style temporal encoding with variable selection
        enc_x = self.positional_encoding(enc_x, temporal_features=temporal_features)

        # Warm-up processing (unchanged)
        enc_bs, enc_seq_len, enc_d_model = enc_x.shape
        if self.configs.warmup_train:
            enc_x = torch.cat([enc_x[:, : int(enc_x.shape[1] / 2), :], enc_x], dim=1)

        # Encoder processing (unchanged)
        hidden_states = self.encoder(enc_x)
        hidden_states = self.encoder_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.enc_2_dec_embedding(hidden_states)
        dec_x = hidden_states

        # Remove warm-up
        dec_x = dec_x[:, -enc_seq_len:, :]

        dec_x_time_series = dec_x[:, :-1, :]
        dec_x_static = dec_x[:, -1, :]

        # Restore outputs (unchanged)
        outputs_time_series = self.time_series_projection(
            dec_x_time_series, feature_order=self.time_series_variables
        )
        (
            outputs_static,
            static_variables_dec_index_start,
            static_variables_dec_index_end,
        ) = self.static_projection(
            dec_x_static,
            feature_order=self.static_variables,
            mode=batch_data_dict['mode'],
        )

        output_dict = {
            'outputs_time_series': outputs_time_series,
            'outputs_static': outputs_static,
            'masked_time_series_index': masked_time_series_index,
            'masked_static_index': masked_static_index,
            "masked_missing_time_series_index": masked_missing_time_series_index,
            "masked_missing_static_index": masked_missing_static_index,
            'static_variables_dec_index_start': static_variables_dec_index_start,
            'static_variables_dec_index_end': static_variables_dec_index_end,
        }

        return output_dict
