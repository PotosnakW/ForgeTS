from types import SimpleNamespace
from torch import nn
import torch

from ..common._base_model import BaseModel
from ..common._modules import PositionalEncoding, _make_causal_token_mask
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
        config.nf = config.hidden_size * patch_num

        self.tokenizer = BaseTokenizer().get_tokenizer(config=config)
        self.input_layer = BaseInputLayer().get_input_layer(config=config)
        self.encoder = BaseEncoder().get_encoder(config=config)
        self.decoder = BaseDecoder().get_decoder(config=config)
        self.output_layer = BaseOutputLayer().get_output_layer(config=config)
    
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

        print(f"before toeknize: {x_enc.shape=}")
        x_enc = self.tokenizer(x=x_enc) # [B, C, n_patch, patch_len]
        print(f"after tokenize: {x_enc.shape=}")
        x_enc = self.input_layer(x=x_enc) # [B, C, n_patch, d_model]
        print(f"after input layer: {x_enc.shape=}")

        x_enc  = x_enc.reshape(
            batch_size * n_channels, patch_num_inp, self.hidden_size
        )

        outputs = self.encoder(
            n_channels = n_channels,
            inputs_embeds = x_enc,
            attention_mask = attention_mask,
        )
        enc_out = outputs.last_hidden_state  # [B*C, n_patch, d_model]

        # standard: [B*C, 1, P, d_model]
        # forking:  [B*C, T, P, d_model]]
        enc_out = self._fs_unfold(enc_out)

        dec_out = self.decoder(enc_out) # [B*C, T, P, d_model]
        dec_out = dec_out.reshape(
            batch_size, n_channels, -1, self.patch_num, self.hidden_size
        ) # [B, C, T, P, d_model]
        output = self.output_layer(dec_out)            # [B, C, H*c_out]

        return output

    def _fs_unfold(self, enc_out: torch.Tensor) -> torch.Tensor:
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
        input_mask = batch["available_mask"].clone()  # [B, L+(T-1)*step_size, C]
        x = x[..., 0] # [B, L+(T-1)*step_size, C]  target only
        x_enc_in = x.permute(0, 2, 1) # [B, C, L+(T-1)*step_size]
        input_mask = input_mask.permute(0, 2, 1) # [B, C, L+(T-1)*step_size]

        forecast = self.model(
            x_enc = x_enc_in,
            available_mask = input_mask,           # [B, C, seq_len]
        )                                          # [B, C, P_total, d_model]

        B, C, T, _ = forecast.shape
        forecast = forecast.reshape(B, C, T, horizon, -1)  # [B, C, T, H, Q]
        forecast = forecast.permute(0, 2, 3, 1, 4)        # [B, T, H, C, Q]
    
        return forecast                                    # [B, T, H, C, Q]
