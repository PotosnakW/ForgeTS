import logging
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
        self.patch_len   = config.patch_len
        self.stride = config.stride

        self.tokenizer = Patching(patch_len=config.patch_len, stride=config.stride)
        self.W_P = nn.Linear(config.patch_len, config.hidden_size)
        self.W_pos = PositionalEncoding(
            pe_type = config.pe_type,
            hidden_size = config.hidden_size,
            learn_pe = config.learn_pe,
        )
        self.dropout = nn.Dropout(config.dropout)

        if config.transformer_backbone in [
            "google/t5-efficient-tiny", 
            "google/t5-efficient-mini",
            "google/t5-efficient-small", 
            "google/t5-efficient-base",
            "google/t5-efficient-large",
        ]:
            self.encoder = self._get_huggingface_transformer(config)
        elif config.transformer_backbone == "patchtst":
            self.encoder = TSTEncoder(config)
        else:
            raise ValueError(
                f"transformer_backbone '{config.transformer_backbone}' not recognised."
            )

    def _get_huggingface_transformer(self, config):
        model_config = T5Config.from_pretrained(config.transformer_backbone)
        for attr in [
            "infini_mixer_type", "infini_channel_exclusion", "layerwise_beta",
            "channelwise_beta", "mlpmixer_hidden_size", "mlpmixer_n_layers",
            "mlpmixer_dropout",
        ]:
            setattr(model_config, attr, getattr(config, attr))
        setattr(model_config, "n_channels", config.n_channels)
        transformer = T5Model(model_config)
        logger.info(f"Randomly initializing {config.transformer_backbone} ({T5Model.__name__}).")
        return transformer.get_encoder()

    def forward(
        self,
        x_enc:          torch.Tensor,
        available_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        x_enc          : [B, C, seq_len]   seq_len = L (standard) or L+(T-1)*s (forking/auto)
        available_mask : [B, seq_len]       1=real timestep, 0=padded/missing (optional)

        Patch attention mask
        ────────────────────
        Unfold available_mask with the same patch_len and stride as tokenizer.
        A patch is masked (0) only if ALL its timesteps are masked.
        Expanded to [B*C, n_patch] — channels share the same time mask.

        returns: [B, C, n_patch, d_model]
        """
        batch_size, n_channels, seq_len = x_enc.shape

        # dynamic n_patch from actual input (handles variable T for fcd_samples=-1)
        n_patch = (seq_len - self.patch_len) // self.stride + 1

        # build attention mask: [B, n_patch] → [B*C, n_patch]
        if available_mask is not None:
            # unfold matches tokenizer: [B, seq_len] → [B, n_patch, patch_len]
            patch_avail    = available_mask.unfold(1, self.patch_len, self.stride)
            patch_mask     = patch_avail.any(dim=-1).float()            # [B, n_patch]
            attention_mask = patch_mask.repeat_interleave(n_channels, dim=0)  # [B*C, n_patch]
        else:
            attention_mask = torch.ones(
                batch_size * n_channels, n_patch, device=x_enc.device
            )

        x_enc  = self.tokenizer(x=x_enc)          # [B, C, n_patch, patch_len]
        x_enc  = self.W_P(x_enc)                  # [B, C, n_patch, d_model]
        x_enc += self.W_pos(x_enc)                # [B, C, n_patch, d_model]
        x_enc  = x_enc.reshape(
            batch_size * n_channels, n_patch, self.hidden_size
        )
        x_enc  = self.dropout(x_enc)

        outputs = self.encoder(
            n_channels     = n_channels,
            inputs_embeds  = x_enc,
            attention_mask = attention_mask,
        )
        enc_out = outputs.last_hidden_state        # [B*C, n_patch, d_model]

        return enc_out.reshape(
            batch_size, n_channels, n_patch, self.hidden_size
        )                                          # [B, C, n_patch, d_model]

class Decoder(nn.Module):                         # FIX: colon, nn.Module
    def __init__(self, config):
        super().__init__()

        patch_num_inp = int((config.input_size - config.patch_len) / config.stride + 1)
        self.patch_num_inp = patch_num_inp

        self.forecast_head = Flatten_Head(
            multivariate_head = config.multivariate_head,
            n_vars = config.n_channels,
            nf = config.hidden_size * patch_num_inp,
            h = config.h,
            c_out = config.c_out,
            head_dropout = config.head_dropout,
        )

        if config.fcd_samples == 1:      # window sampling only
            self.decode = self._decode_standard
        else:                             # >1 or -1 → forking
            self.decode = self._decode_forking

    def _decode_standard(self, enc_out: torch.Tensor) -> torch.Tensor:
        """
        enc_out : [B, C, P_std, d_model]
        returns : [B, C, 1, H*c_out]   T=1 so forward is identical for both modes
        """
        B, C = enc_out.shape[:2]
        flat = enc_out.reshape(B, C, -1)           # [B, C, P_std*d_model]
        pred = self.forecast_head(flat)            # [B, C, H*c_out]
        return pred.unsqueeze(2)                   # [B, C, 1, H*c_out]  T=1 unifies both modes

    def _decode_forking(self, enc_out: torch.Tensor) -> torch.Tensor:
        """
        enc_out : [B, C, P_std+T-1, d_model]   T inferred from enc_out shape
        returns : [B, C, T, H*c_out]

        Slides a P_std-patch window over P_total encoder patches → T predictions.
        T is derived at runtime from enc_out so fcd_samples=-1 (variable T) works.
        step=1 patch is correct: fork_sequences already spaced raw blocks by
        stride timesteps, so consecutive patch windows are exactly 1 patch apart.
        """
        B, C, patch_num, d = enc_out.shape

        windows = (
            enc_out
            .unfold(dimension=2, size=self.patch_num_inp, step=1)   # [B, C, T, d, P]
            .permute(0, 1, 2, 4, 3)                # [B, C, T, P, d]
            .contiguous()
        )
        flat = windows.reshape(B, C, -1, self.patch_num_inp * d)     # [B, C, T, P_std*d_model]
        return self.forecast_head(flat)             # [B, C, T, H*c_out]

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        return self.decode(enc_out)
        # standard: [B, C, 1, H*c_out]
        # forking:  [B, C, T, H*c_out]

class MOMENT(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__()

        if isinstance(config, dict):
            config = SimpleNamespace(**config)

        assert (config.input_size - config.patch_len) % config.stride == 0, (
            f"(input_size - patch_len) % stride must be 0, got "
            f"({config.input_size} - {config.patch_len}) % {config.stride} = "
            f"{(config.input_size - config.patch_len) % config.stride}"
        )
        config.patch_len = min(config.input_size, config.patch_len)
        config.c_out = config.loss.outputsize_multiplier

        self.fcd_samples = config.fcd_samples
        self.h = config.h
        self.n_channels = config.n_channels

        self.encoder = Encoder(config=config)
        self.decoder = Decoder(config=config)

        self.revin = config.revin
        if config.revin:
            self.revin_layer = RevINMultivariate(
                num_features = config.n_channels,
                affine = config.revin_affine,
                subtract_last = config.revin_subtract_last,
            )

    def forward(
        self,
        x: torch.Tensor,   # [B, seq_len, C, V]
        mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:

        x = x[..., 0]                    # [B, seq_len, C]  target only
        x_enc_in   = x.permute(0, 2, 1)           # [B, C, seq_len]

        if self.revin:
            x_enc_in = x_enc_in.permute(0, 2, 1)  # [B, seq_len, C]
            x_enc_in = self.revin_layer(x_enc_in, "norm")
            x_enc_in = x_enc_in.permute(0, 2, 1)  # [B, C, seq_len]

        enc_out  = self.encoder(
            x_enc          = x_enc_in,
            available_mask = input_mask,           # [B, seq_len] from fork_sequences or caller
        )                                          # [B, C, P_total, d_model]
        forecast = self.decoder(enc_out=enc_out)
        # both modes: [B, C, T, H*c_out]  — T=1 for standard, T=fcd_samples for forking

        # RevIN denorm:
        if self.revin:
            B, C, T, Hc = forecast.shape
            forecast = forecast.permute(0, 2, 1, 3).reshape(B * T, C, Hc)  # [B*T, C, H*c_out]
            forecast = forecast.permute(0, 2, 1)                            # [B*T, H*c_out, C]
            forecast = self.revin_layer(forecast, "denorm")                 # [B*T, H*c_out, C]
            forecast = forecast.permute(0, 2, 1).reshape(B, T, C, Hc)      # [B, T, C, H*c_out]
            forecast = forecast.permute(0, 2, 1, 3)                        # [B, C, T, H*c_out]

        B, C, T, _ = forecast.shape
        forecast = forecast.reshape(B, C, T, self.h, -1)   # [B, C, T, H, c_out]
        forecast = forecast.permute(0, 2, 3, 4, 1)         # [B, T, H, c_out, C]
        forecast = forecast.reshape(B * T, self.h, -1)     # [B*T, H, C*c_out]

        return forecast  # [B*T, H, C*c_out]
