import logging
import warnings
from typing import Optional
from types import SimpleNamespace

import torch
from torch import nn

from ..common._base_model import BaseModel
from ..common._modules import RevINMultivariate, Flatten_Head, Patching, PositionalEncoding
from ..encoders.t5_encoder import T5Model
from ..encoders.tst_encoder import TSTEncoder

from transformers import T5Config

logger = logging.getLogger(__name__)


class Encoder(nn.Module): 
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.patch_len = config.patch_len

        self.revin = config.revin
        if config.revin:
            self.revin_layer = RevINMultivariate(
                num_features=config.n_series, 
                affine=config.revin_affine,
                subtract_last=config.revin_subtract_last,
            )

        self.padding_patch = config.padding_patch
        patch_num = int((config.input_size - config.patch_len) / config.stride + 1)
        if config.padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, config.stride))
            patch_num += 1
        self.patch_num = patch_num

        self.tokenizer = Patching(
            patch_len=config.patch_len, 
            stride=config.stride,
        )

        self.W_P = nn.Linear(
            config.patch_len, config.hidden_size
        )  # Eq 1: projection of feature vectors onto a d-dim vector space

        # Positional encoding
        self.W_pos = PositionalEncoding(
            pe_type=config.pe_type,
            hidden_size=config.hidden_size,
            learn_pe=config.learn_pe,
        )
        # Residual dropout
        self.dropout = nn.Dropout(config.dropout)

        # Transformer backbone
        if config.transformer_backbone in [
            "google/t5-efficient-tiny",
            "google/t5-efficient-mini",
            "google/t5-efficient-small",
            "google/t5-efficient-base",
            "google/t5-efficient-large",
        ]:
            self.encoder = self._get_huggingface_transformer(config)
        elif config.transformer_backbone == 'patchtst':
            self.encoder = TSTEncoder(config)
        else:
            raise Exception("transformer_bacbone is not recognized. Must be one of ['patchtst', 'google/t5-efficient-{tiny, mini, small, base, large}']")

        # Prediction Head
        self.head = Flatten_Head(
            multivariate_head=config.multivariate_head,
            n_vars=config.n_series,
            nf=config.hidden_size * patch_num,
            h=config.h,
            c_out=config.c_out,
            head_dropout=config.head_dropout,
        )

    def _get_huggingface_transformer(self, config):
            
        model_config = T5Config.from_pretrained(
            config.transformer_backbone)

        setattr(model_config, 'infini_mixer_type', config.infini_mixer_type)
        setattr(model_config, 'infini_channel_exclusion', config.infini_channel_exclusion)
        setattr(model_config, 'layerwise_beta', config.layerwise_beta)
        setattr(model_config, 'channelwise_beta', config.channelwise_beta)
        setattr(model_config, 'n_channels', config.n_series)
        setattr(model_config, 'mlpmixer_hidden_size', config.mlpmixer_hidden_size)
        setattr(model_config, 'mlpmixer_n_layers', config.mlpmixer_n_layers)
        setattr(model_config, 'mlpmixer_dropout', config.mlpmixer_dropout)
      
        transformer_backbone = T5Model(model_config)
        logging.info(f"Initializing randomly initialized\
                       transformer from {config.transformer_backbone}.  ModelClass: {T5Model.__name__}.")
        
        transformer_backbone = transformer_backbone.get_encoder()
        
        return transformer_backbone

    def forward(self, 
                x_enc : torch.Tensor,
                **kwargs):
        """
        x_enc : [batch_size x n_channels x seq_len]
        input_mask : [batch_size x seq_len]
        """

        batch_size, n_channels, seq_len = x_enc.shape
        attention_mask = torch.ones(batch_size*n_channels, self.patch_num, device=x_enc.device) # no masking, 1==available

        # Normalization (applied over axis=1)
        if self.revin:
            x_enc = x_enc.permute(0, 2, 1) # [batch_size x seq_len x n_channel]
            x_enc = self.revin_layer(x_enc, "norm")
            x_enc = x_enc.permute(0, 2, 1) # [batch_size x n_channel x seq_len]
        
        # Patching
        if self.padding_patch == "end":
            x_enc = self.padding_patch_layer(x_enc) 
        x_enc = self.tokenizer(x=x_enc) # [batch_size x n_channels x n_patch x patch_len]

        # Embeddings
        x_enc = self.W_P(x_enc) # [batch_size x n_channels x n_patch x d_model]
        x_enc += self.W_pos(x_enc) # [batch_size x n_channels x n_patch x d_model]
        
        x_enc = x_enc.reshape(
            (batch_size * n_channels, self.patch_num, self.hidden_size)) # [batch_size*n_channels, n_patch, d_model]
        x_enc = self.dropout(x_enc) # [batch_size*n_channels, n_patch, d_model]

        # Encoder
        outputs = self.encoder(
            n_channels=n_channels,
            inputs_embeds=x_enc, 
            attention_mask=attention_mask, 
        ) 
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape(
            (batch_size, n_channels, self.patch_num, self.hidden_size)
        ) # [batch_size, n_channels, n_patch, d_model]

        # Decoder
        dec_out = self.head(enc_out) # [batch_size, n_channels, horizon*c_out]
        
        # De-Normalization
        if self.revin:
            dec_out = dec_out.permute(0, 2, 1) # [batch_size x horizon*c_out x n_channel]
            dec_out = self.revin_layer(dec_out, "denorm")
            dec_out = dec_out.permute(0, 2, 1) # [batch_size x n_channel x horizon*c_out]

        return dec_out

class MOMENT(BaseModel):
    def __init__(self, config: Namespace | dict, **kwargs: dict):
        super().__init__()  # Added missing super().__init__() call

        config = SimpleNamespace(**config)

        patch_len = min(config.input_size + config.stride, config.patch_len)
        config['patch_len'] = config.patch_len
        config['c_out'] = config.loss.outputsize_multiplier

        self.h = config.h
        self.n_channels = config.n_channels
        self.model = Encoder(config=config)

    def forward(self, 
                x : torch.Tensor,  # [batch_size (B), input_size (L), n_channels (C)] #, n_covariates (I)]
                mask : torch.Tensor = None, 
                input_mask : torch.Tensor = None,
                **kwargs):

        batch_size = x.shape[0]
        x_enc = x.permute(0, 2, 1) # [batch_size (B), n_channels (C), input_size (L)]
        forecast = self.model(x_enc=x_enc) # [batch_size, n_channels, horizon*c_out]

        forecast = forecast.view(batch_size, self.n_channels, self.h, -1) # [batch_size, n_channels, horizon, c_out]
        forecast = forecast.permute(0, 2, 3, 1).reshape(batch_size, self.h, -1) # [batch_size, horizon, c_out*n_channels] 
        # output is expected in this shape. tsmixer and other neuralforecast multivariate models' decoder output is already in shape # [batch_size, horizon*c_out, n_channels] so skipping to forecast.reshape(batch_size, self.h, -1) is valid for those models. 

        return forecast
