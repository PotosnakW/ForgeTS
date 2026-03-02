import torch 
import torch.nn as nn

from ..common._infini import MultiheadAttention
from ..common._modules import Transpose


class TSTEncoder(nn.Module):
    """
    TSTEncoder
    """
    def __init__(
        self,
        config,
    ):
        super().__init__()

        if config.infini_mixer_type == 'betas':
            if config.layerwise_beta:
                beta = None
            else:
                beta = nn.Parameter(torch.rand((1, 1, config.n_heads, 1, 1))*1e-2)
                # Adjust the values to ensure they sum to 0
                with torch.no_grad():
                    beta -= beta.mean(dim=2, keepdim=True)
        else:
            beta=None

        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    config=config,
                    beta=beta,
                )
                for i in range(config.n_layers)
            ]
        )
        self.res_attention = config.res_attention

    def forward(
        self,
        src: torch.Tensor,
        n_channels: int,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(
                    src=output,
                    n_channels=n_channels,
                    prev=scores,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )
            return output
        else:
            for mod in self.layers:
                output = mod(
                    src=output, 
                    n_channels=n_channels,
                    key_padding_mask=key_padding_mask, 
                    attn_mask=attn_mask
                )
            return output

class TSTEncoderLayer(nn.Module):
    """
    TSTEncoderLayer
    """
    def __init__(
        self,
        config,
        beta: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Multi-Head attention
        self.res_attention = config.res_attention
        self.self_attn = MultiheadAttention(
            config=config,
            beta=beta,
        )

        # Add & Norm
        self.dropout_attn = nn.Dropout(config.dropout)
        if "batch" in config.norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(config.hidden_size), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(config.hidden_size)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_size, config.linear_hidden_size, bias=True), # bias hard-coded in PatchTST
            get_activation_fn(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.linear_hidden_size, config.hidden_size, bias=True), # bias hard-coded in PatchTST
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(config.dropout)
        if "batch" in config.norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(config.hidden_size), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(config.hidden_size)

        self.pre_norm = config.pre_norm
        self.store_attn = config.store_attn

    def forward(
        self,
        src: torch.Tensor,
        n_channels: int,
        prev: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):  # -> Tuple[torch.Tensor, Any]:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(
                n_channels=n_channels,
                Q=src,
                K=src,
                V=src,
                prev=prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            src2, attn = self.self_attn(
                n_channels=n_channels,
                Q=src,
                K=src,
                V=src,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src
    