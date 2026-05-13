from types import SimpleNamespace
from torch import nn
import torch

from ..common._base_model import BaseModel
from ..tokenizers._base_tokenizer import BaseTokenizer
from ..input_layers._base_input_layer import BaseInputLayer
from ..encoders._base_encoder import BaseEncoder
from ..decoders._base_decoder import BaseDecoder
from ..output_layers._base_output_layer import BaseOutputLayer


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        config.nf = config.hidden_size
        self.c_out = config.c_out

        self.tokenizer = BaseTokenizer().get_tokenizer(config=config)
        self.input_layer = BaseInputLayer().get_input_layer(config=config)
        self.encoder = BaseEncoder().get_encoder(config=config)
        self.decoder = BaseDecoder().get_decoder(config=config)
        self.output_layer = BaseOutputLayer().get_output_layer(config=config)
    
    def forward(self, x_enc, horizon, fcd_samples, available_mask=None, **kwargs):
        batch_size, n_channels, seq_len = x_enc.shape

        x_enc = self.tokenizer(x=x_enc)
        x_enc = self.input_layer(x=x_enc) 

        x = x_enc.reshape(batch_size * n_channels, seq_len, self.hidden_size)

        enc_out = self.encoder(inputs_embeds=x, n_channels=n_channels) # [B*C, seq_len, hidden_size]
    
        assert fcd_samples > 0, f"fcd_samples must be resolved before Model.forward, got {fcd_samples}"
        enc_out = enc_out[:, -fcd_samples:, :]
        enc_out = enc_out.unsqueeze(2)  # [B*C, fcd_samples, 1, hidden_size]

        dec_out = self.decoder(enc_out)                          # [B*C, T, 1, hidden_size]
        dec_out = dec_out.reshape(batch_size, n_channels, *dec_out.shape[1:])
        output = self.output_layer(dec_out)                     # [B, C, T, H*c_out]
        output = output.reshape(batch_size, n_channels, fcd_samples, horizon, self.c_out)  # [B, C, T, K*O, c_out]
    
        return output

class RNN(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        if isinstance(config, dict):
            config = SimpleNamespace(**config)

        config.c_out = self.loss_fn.outputsize_multiplier
    
        self.model = Model(config=config)

    def forward(
        self,
        batch,
    ) -> torch.Tensor:

        # TODO @wpotosna Extend for covariates

        horizon = batch["horizon"]

        x = batch["insample_y"].clone() # [B, L+(T-1)*step_size, C, 1+Vh]
        input_mask = batch["available_mask"].clone()  # [B, L+(T-1)*step_size, C]
        x = x[..., 0] # [B, L+(T-1)*step_size, C]  target only
        x_enc_in = x.permute(0, 2, 1) # [B, C, L+(T-1)*step_size]
        input_mask = input_mask.permute(0, 2, 1) # [B, C, L+(T-1)*step_size]

        forecast = self.model(
            x_enc = x_enc_in,
            horizon = horizon,
            fcd_samples = batch.get("fcd_samples"),
            available_mask = input_mask,           # [B, C, seq_len]
        )                                          # [B, C, P_total, d_model]
        forecast = forecast.permute(0, 2, 3, 1, 4)        # [B, T, H, C, Q]
    
        return forecast                                    # [B, T, H, C, Q]
