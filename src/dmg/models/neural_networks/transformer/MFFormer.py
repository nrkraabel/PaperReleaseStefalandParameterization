import torch
from models.neural_networks.transformer.features_embedding import (
    StaticDecEmbedding,
    StaticEncEmbedding,
    TimeSeriesDecEmbedding,
    TimeSeriesEncEmbedding,
)
from models.neural_networks.transformer.mask import MaskGenerator
from models.neural_networks.transformer.positional_encoding import PositionalEncoding
from models.neural_networks.transformer.transformer_layers import TransformerBackbone
from torch import nn

# from MFFormer.utils.validation_tools import Validator


# define the Multiple Features for Time Series Transformer
class Model(nn.Module):
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

        # # norm the layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        # self.encoder_norm = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(embed_dim), Transpose(1, 2))
        # self.decoder_norm = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(embed_dim), Transpose(1, 2))

        # time series embedding
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

        # positional encoding
        self.positional_encoding = PositionalEncoding(
            embed_dim, dropout=configs.dropout
        )

        # define the mask function
        self.mask_generator = MaskGenerator(
            configs.mask_ratio_time_series,
            mask_ratio_static=configs.mask_ratio_static,
            min_window_size=configs.min_window_size,
            max_window_size=configs.max_window_size,
        )

        # define the encoder
        self.encoder = TransformerBackbone(
            embed_dim, configs.num_enc_layers, d_ffd, configs.num_heads, configs.dropout
        )

        # define the decoder
        self.enc_2_dec_embedding = nn.Linear(embed_dim, embed_dim, bias=True)
        self.decoder = TransformerBackbone(
            embed_dim, configs.num_dec_layers, d_ffd, configs.num_heads, configs.dropout
        )

        # define the projection layer
        self.time_series_projection = TimeSeriesDecEmbedding(
            configs.time_series_variables,
            embed_dim,
            dropout=configs.dropout,
            add_input_noise=configs.add_input_noise,
        )
        self.static_projection = StaticDecEmbedding(
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
            # nn.init.kaiming_uniform_(layer.weight, a=0.02)
            nn.init.uniform_(
                layer.weight, -self.configs.init_weight, self.configs.init_weight
            )
            nn.init.uniform_(
                layer.bias, -self.configs.init_bias, self.configs.init_bias
            )

        nn.init.uniform_(
            self.positional_encoding.position_embedding,
            -self.configs.init_weight,
            self.configs.init_weight,
        )

        layers_to_init = [
            *self.time_series_embedding.embeddings1.values(),
            *self.time_series_embedding.embeddings2.values(),
            *self.static_embedding.numerical_embeddings1.values(),
            *self.static_embedding.numerical_embeddings2.values(),
            *self.time_series_projection.embeddings1.values(),
            *self.time_series_projection.embeddings2.values(),
            *self.static_projection.numerical_embeddings1.values(),
            *self.static_projection.numerical_embeddings2.values(),
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

        batch_x = batch_data_dict['batch_x']  # [batch_size, seq_len, num_features]
        batch_c = batch_data_dict['batch_c']
        masked_time_series_index = batch_data_dict['batch_time_series_mask_index']
        masked_static_index = batch_data_dict['batch_static_mask_index']

        # generate the mask
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

            # missing data mask
            masked_missing_time_series_index = torch.isnan(batch_x)
            masked_missing_static_index = torch.isnan(batch_c)
            # replace the missing value with 0
            batch_x = batch_x.masked_fill(masked_missing_time_series_index, 0)
            batch_c = batch_c.masked_fill(masked_missing_static_index, 0)

            # combine the mask
            masked_time_series_index = (
                masked_time_series_index | masked_missing_time_series_index
            )
            masked_static_index = masked_static_index | masked_missing_static_index

        else:
            unmasked_time_series_index, masked_time_series_index = None, None
            unmasked_static_index, masked_static_index = None, None

        # # replace the masked value with 0
        # batch_x = batch_x.masked_fill(masked_time_series_index, 0)
        # batch_c = batch_c.masked_fill(masked_static_index, 0)

        # features embedding, [batch_size, seq_len, d_model]
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

        # concat the time series and static features
        enc_x = torch.cat([enc_x, enc_c[:, None, :]], dim=1)

        # add the positional encoding
        enc_x = self.positional_encoding(enc_x)

        # warm-up the encoder
        enc_bs, enc_seq_len, enc_d_model = enc_x.shape
        if self.configs.warmup_train:
            enc_x = torch.cat([enc_x[:, : int(enc_x.shape[1] / 2), :], enc_x], dim=1)

        # encoder
        hidden_states = self.encoder(enc_x)
        hidden_states = self.encoder_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.enc_2_dec_embedding(hidden_states)

        if self.configs.num_dec_layers > 0:
            # decoder
            dec_x = self.decoder(hidden_states)
            dec_x = self.decoder_norm(dec_x)
            dec_x = self.dropout(dec_x)
        else:
            dec_x = hidden_states

        # remove the warm-up
        dec_x = dec_x[:, -enc_seq_len:, :]

        dec_x_time_series = dec_x[:, :-1, :]  # [batch_size, seq_len, d_model]
        dec_x_static = dec_x[:, -1, :]  # [batch_size, d_model]

        # restore
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
            # 'unmasked_time_series_index': unmasked_time_series_index.to(outputs_time_series.device),
            # 'unmasked_static_index': unmasked_static_index.to(outputs_time_series.device),
            "masked_missing_time_series_index": masked_missing_time_series_index,
            "masked_missing_static_index": masked_missing_static_index,
            'static_variables_dec_index_start': static_variables_dec_index_start,
            'static_variables_dec_index_end': static_variables_dec_index_end,
        }

        return output_dict
