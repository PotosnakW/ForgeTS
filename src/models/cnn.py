from types import SimpleNamespace
from torch import nn
import torch

from common._base_model import BaseModel
from common._modules import RevIN
from encoders._base_encoder import BaseEncoder
from decoders._base_decoder import BaseDecoder
from output_layers._base_output_layer import BaseOutputLayer


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        config.nf = config.hidden_size

        self.W_P = nn.Linear(1, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.encoder = BaseEncoder().get_encoder(config=config)
        self.decoder = BaseDecoder().get_decoder(config=config)
        self.output_layer = BaseOutputLayer().get_output_layer(config=config)
    
    def forward(self, x_enc, available_mask=None, **kwargs):
        batch_size, n_channels, seq_len = x_enc.shape

        x_enc = x_enc.reshape(batch_size * n_channels, seq_len, 1) # [B*C, seq_len, 1]
        x_enc = x_enc.permute(0, 2, 1) # [B*C, 1, seq_len]
        
        enc_out = self.encoder(x=x_enc, n_channels=n_channels)
        enc_out = enc_out.unsqueeze(2)         # [B*C, seq_len, 1, hidden_size]

        dec_out = self.decoder(enc_out)
        dec_out = dec_out.reshape(batch_size, n_channels, seq_len, 1, self.hidden_size)
        output  = self.output_layer(dec_out)   # [B, C, seq_len, H*c_out]

        return output

class CNN(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        if isinstance(config, dict):
            config = SimpleNamespace(**config)

        config.c_out = self.loss_fn.outputsize_multiplier

        self.revin = config.revin
        if config.revin:
            self.revin_layer = RevIN(
                affine        = config.revin_affine,
                subtract_last = config.revin_subtract_last,
            )

        self.model = Model(config=config)

    def forward(
        self,
        batch,
    ) -> torch.Tensor:

        # TODO @wpotosna Extend for covariates

        horizon = getattr(self.mcfg, "horizon_override", None) or int(batch["horizon"][0].item())

        x = batch["insample_y"].clone() # [B, L+(T-1)*step_size, C, 1+Vh]
        input_mask = batch["available_mask"].clone()  # [B, L+(T-1)*step_size, C]
        x = x[..., 0] # [B, L+(T-1)*step_size, C]  target only
        x_enc_in = x.permute(0, 2, 1) # [B, C, L+(T-1)*step_size]
        input_mask = input_mask.permute(0, 2, 1) # [B, C, L+(T-1)*step_size]

        if self.revin:
            x_enc_in = x_enc_in.permute(0, 2, 1)  # [B, seq_len, C]
            x_enc_in = self.revin_layer(x_enc_in, "norm")
            x_enc_in = x_enc_in.permute(0, 2, 1)  # [B, C, seq_len]

        forecast = self.model(
            x_enc = x_enc_in,
            available_mask = input_mask,           # [B, C, seq_len]
        )                                          # [B, C, P_total, d_model]

        # RevIN denorm:
        if self.revin:
            B, C, T, Hc = forecast.shape
            forecast = forecast.permute(0, 2, 1, 3).reshape(B * T, C, Hc)  # [B*T, C, H*c_out]
            forecast = forecast.permute(0, 2, 1)                            # [B*T, H*c_out, C]
            forecast = self.revin_layer(forecast, "denorm")                 # [B*T, H*c_out, C]
            forecast = forecast.permute(0, 2, 1).reshape(B, T, C, Hc)      # [B, T, C, H*c_out]
            forecast = forecast.permute(0, 2, 1, 3)                        # [B, C, T, H*c_out]

        B, C, T, _ = forecast.shape
        forecast = forecast.reshape(B, C, T, horizon, -1)  # [B, C, T, H, Q]
        forecast = forecast.permute(0, 2, 3, 1, 4)        # [B, T, H, C, Q]
    
        return forecast                                    # [B, T, H, C, Q]
