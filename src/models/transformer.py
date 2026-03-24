from types import SimpleNamespace
from torch import nn
import torch

from common._base_model import BaseModel
from common._modules import RevIN, Patching, PositionalEncoding, _make_causal_token_mask
from encoders._base_encoder import BaseEncoder
from decoders._base_decoder import BaseDecoder
from output_layers._base_output_layer import BaseOutputLayer


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.patch_len = config.patch_len
        self.stride = config.stride
        patch_num = int((config.context_length - config.patch_len) / config.stride + 1)
        self.patch_num = patch_num
        config.nf = config.hidden_size * patch_num

        self.tokenizer = Patching(patch_len=config.patch_len, stride=config.stride)
        self.W_P = nn.Linear(config.patch_len, config.hidden_size)
        self.W_pos = PositionalEncoding(
            pe_type = config.pe_type,
            hidden_size = config.hidden_size,
            learn_pe = config.learn_pe,
        )
        self.dropout = nn.Dropout(config.dropout)

        self.encoder = BaseEncoder().get_encoder(config=config)
        self.decoder = BaseDecoder().get_decoder(config=config)
        self.output_layer = BaseOutputLayer().get_output_layer(config=config)

        if config.fcd_samples == 1:      # window sampling only
            self.fs_ws = self._ws_output
        else:                             # >1 or -1 → forking
            self.fs_ws = self._fs_output
    
    def forward(
        self,
        x_enc:          torch.Tensor,
        available_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        x_enc          : [B, C, seq_len]   seq_len = L (standard) or L+(T-1)*s (forking/auto)
        available_mask : [B, C, seq_len]       1=real timestep, 0=padded/missing (optional)

        Patch attention mask
        ────────────────────
        Unfold available_mask with the same patch_len and stride as tokenizer.
        A patch is masked (0) only if ALL its timesteps are masked.
        Expanded to [B*C, n_patch] — channels share the same time mask.

        returns: [B, C, n_patch, d_model]
        """
        batch_size, n_channels, seq_len = x_enc.shape

        # dynamic n_patch from actual input (handles variable T for fcd_samples=-1)
        patch_num_inp = (seq_len - self.patch_len) // self.stride + 1

        # build attention mask: [B, C, n_patch]
        if available_mask is not None:
            patch_avail = available_mask.unfold(-1, self.patch_len, self.stride)
            key_padding_mask = patch_avail.any(dim=-1).float() # [B, C, n_patch]
        else:
            key_padding_mask = torch.ones(
                batch_size, n_channels, patch_num_inp
            )

        attention_mask = _make_causal_token_mask(key_padding_mask=key_padding_mask, device=x_enc.device) # [B, C, 1, n_patch, n_patch]
        attention_mask = attention_mask.reshape(batch_size * n_channels, 1, patch_num_inp, patch_num_inp) # [B * C, 1, n_patch, n_patch]

        x_enc  = self.tokenizer(x=x_enc)          # [B, C, n_patch, patch_len]
        x_enc  = self.W_P(x_enc)                  # [B, C, n_patch, d_model]
        x_enc += self.W_pos(x_enc)                # [B, C, n_patch, d_model]
        x_enc  = x_enc.reshape(
            batch_size * n_channels, patch_num_inp, self.hidden_size
        )
        x_enc  = self.dropout(x_enc)

        outputs = self.encoder(
            n_channels = n_channels,
            inputs_embeds = x_enc,
            attention_mask = attention_mask,
        )
        enc_out = outputs.last_hidden_state  # [B*C, n_patch, d_model]

        # standard: [B*C, 1, P, d_model]
        # forking:  [B*C, T, P, d_model]]
        enc_out = self.fs_ws(enc_out)

        dec_out = self.decoder(enc_out) # [B*C, T, P, d_model]
        dec_out = dec_out.reshape(
            batch_size, n_channels, -1, self.patch_num, self.hidden_size
        ) # [B, C, T, P, d_model]
        output = self.output_layer(dec_out)            # [B, C, H*c_out]

        return output

    def _ws_output(self, enc_out: torch.Tensor) -> torch.Tensor:
        """
        enc_out : [B*C, P_std, d_model]
        returns : [B*C, 1, P_std, d_model]   T=1 so forward is identical for both modes
        """
        return enc_out.unsqueeze(1)                   # [B, C, 1, P_std, d_model]  T=1 unifies both modes

    def _fs_output(self, enc_out: torch.Tensor) -> torch.Tensor:
        """
        enc_out : [B*C, P_std+T-1, d_model]   T inferred from enc_out shape
        returns : [B*C, T, H*c_out]

        Slides a P_std-patch window over P_total encoder patches → T predictions.
        T is derived at runtime from enc_out so fcd_samples=-1 (variable T) works.
        step=1 patch is correct: fork_sequences already spaced raw blocks by
        stride timesteps, so consecutive patch windows are exactly 1 patch apart.
        """
        _, patch_num_inp, _ = enc_out.shape
        enc_out = (
            enc_out
            .unfold(dimension=1, size=self.patch_num, step=1)  # [B*C, T, d_model, patch_num_inp]
            .permute(0, 1, 3, 2) # [B*C, T, patch_num_inp, d_model]
            .contiguous()
        )
        return enc_out

class Transformer(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        if isinstance(config, dict):
            config = SimpleNamespace(**config)

        assert (config.context_length - config.patch_len) % config.stride == 0, (
            f"(context_length - patch_len) % stride must be 0, got "
            f"({config.context_length} - {config.patch_len}) % {config.stride} = "
            f"{(config.context_length - config.patch_len) % config.stride}"
        )
        config.patch_len = min(config.context_length, config.patch_len)
        config.c_out = config.loss.outputsize_multiplier

        self.fcd_samples = config.fcd_samples
        self.h = config.h

        self.revin = config.revin
        if config.revin:
            self.revin_layer = RevIN(
                affine = config.revin_affine,
                subtract_last = config.revin_subtract_last,
            )

        self.model = Model(config=config)

    def forward(
        self,
        batch,
    ) -> torch.Tensor:

        # TODO @wpotosna Extend MICA for covariates

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
        forecast = forecast.reshape(B, C, T, self.h, -1)  # [B, C, T, H, Q]
        forecast = forecast.permute(0, 2, 3, 1, 4)        # [B, T, H, C, Q]
    
        return forecast                                    # [B, T, H, C, Q]
