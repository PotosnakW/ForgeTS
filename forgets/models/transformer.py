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
        self.patch_len = config.patch_len
        self.stride = config.stride
        patch_num = int((config.context_len - config.patch_len) / config.stride + 1)
        self.patch_num = patch_num
        config.nf = config.nf
        self.c_out = config.c_out
        self.decode_fcd_size = getattr(config, "decode_fcd_size", -1)

        self.tokenizer = BaseTokenizer().get_tokenizer(config=config)
        self.input_layer = BaseInputLayer().get_input_layer(config=config)
        self.encoder = BaseEncoder().get_encoder(config=config)
        self.decoder = BaseDecoder().get_decoder(config=config)
        self.output_layer = BaseOutputLayer().get_output_layer(config=config)
    
    def forward(
        self,
        x_enc: torch.Tensor,
        horizon: int,
        fcd_samples: int,
        available_mask: torch.Tensor = None,
        channel_mask: torch.Tensor = None,
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
            key_padding_mask = patch_avail.any(dim=-1).float()         # [B, C, T]
        else:
            key_padding_mask = torch.ones(
                batch_size, n_channels, patch_num_inp, device=x_enc.device
            )

        x_enc = self.tokenizer(x=x_enc)                              # [B, C, T, patch_len]
        x_enc = self.input_layer(x=x_enc)                            # [B, C, T, d]
        x_enc = x_enc.reshape(batch_size * n_channels, patch_num_inp, self.hidden_size)

        enc_out = self.encoder(
            n_channels = n_channels,
            x = x_enc,
            key_padding_mask = key_padding_mask,
            channel_mask = channel_mask,
        )

            # get raw decoder output — either all at once, or chunked internally
        if self.decode_fcd_size == -1:
            dec_out = self._decode_full(enc_out, key_padding_mask, horizon)
        else:
            dec_out = self._decode_chunked(enc_out, key_padding_mask, horizon, fcd_samples)

        dec_out = dec_out.reshape(batch_size, n_channels, fcd_samples, *dec_out.shape[2:])
        output = self.output_layer(dec_out)
        output = output.reshape(batch_size, n_channels, fcd_samples, -1, self.c_out)
        output = output[:, :, :, :horizon, :]

        return output

    def _decode_full(self, enc_out, key_padding_mask, horizon) -> torch.Tensor:
        """Compute-efficient: one decoder call over all origins. Current/original behavior."""
        enc_windows = self._fs_unfold(enc_out)   # [B*C, T, P, d] — full unfold, one shot
        return self.decoder(x=enc_windows, key_padding_mask=key_padding_mask, horizon=horizon)

    def _decode_chunked(self, enc_out, key_padding_mask, horizon, fcd_samples) -> torch.Tensor:
        """Memory-efficient: loops decoder calls over origin windows of decode_chunk_size."""
        outs = []
        for c0 in range(0, fcd_samples, self.decode_fcd_size):
            c1 = min(c0 + self.decode_fcd_size, fcd_samples)
            enc_window = self._fs_unfold(enc_out, window_offset=c0, n_windows=c1 - c0)
            outs.append(self.decoder(x=enc_window, key_padding_mask=key_padding_mask, horizon=horizon))
        return torch.cat(outs, dim=1)   # concat along origin dim, still raw dec_out shape

    def _fs_unfold(self, enc_out, window_offset=0, n_windows=None):
        """
        enc_out : [B*C, patch_num_inp, d]
        returns : [B*C, n_windows, patch_num, d]   (n_windows = all origins if unspecified)

        Slicing happens before .contiguous() so chunked calls only materialize
        their own window range, not the full origin dimension.
        """
        full = enc_out.unfold(dimension=1, size=self.patch_num, step=1).permute(0, 1, 3, 2)
        if n_windows is None:
            n_windows = full.shape[1]
        return full[:, window_offset:window_offset + n_windows].contiguous()

class Transformer(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        if isinstance(config, dict):
            config = SimpleNamespace(**config)

        assert config.context_len != -1, (
            "Transformer requires a fixed context_length — "
            "it partitions input into patches of size patch_len and cannot "
            "operate on variable-length context. Set context_len in your config."
        )

        assert (config.context_len - config.patch_len) % config.stride == 0, (
            f"(context_length - patch_len) % stride must be 0, got "
            f"({config.context_len} - {config.patch_len}) % {config.stride} = "
            f"{(config.context_len - config.patch_len) % config.stride}"
        )
        config.patch_len = min(config.context_len, config.patch_len)
        config.c_out = self.loss_fn.outputsize_multiplier

        self.fcd_samples = config.fcd_samples

        self.model = Model(config=config)

    def forward(
        self,
        batch,
    ) -> torch.Tensor:

        # TODO @wpotosna Extend MICA for covariates

        horizon = batch["horizon"]
        x = batch["insample_y"].clone() # [B, L+(T-1)*step_size, C, 1+Vh]
        available_mask = batch["available_mask"].clone()  # [B, L+(T-1)*step_size, C]
        channel_mask = batch["channel_mask"].clone()  # [B, L+(T-1)*step_size, C]

        x = x[..., 0] # [B, L+(T-1)*step_size, C]  target only
        x_enc_in = x.permute(0, 2, 1) # [B, C, L+(T-1)*step_size]
        available_mask = available_mask.permute(0, 2, 1) # [B, C, L+(T-1)*step_size]

        forecast = self.model(
            x_enc = x_enc_in,
            horizon=horizon, 
            fcd_samples = batch.get("fcd_samples"),
            available_mask = available_mask, # [B, C, seq_len]
            channel_mask = channel_mask, # [B, C]
        )
        forecast = forecast.permute(0, 2, 3, 1, 4)        # [B, T, H, C, Q]
    
        return forecast                                    # [B, T, H, C, Q]
