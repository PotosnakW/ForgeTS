import torch
import torch.nn as nn

from ..common._modules import PositionalEncoding


class LinearProjectionLayer(nn.Module):
    """
    Shared linear projection across all channels.
    Flattens [B, C, P, d_model] → projects to [B, C, H*c_out].
    Efficient and parameter-efficient — same weights for all channels.
    """
    def __init__(self, config):
        super().__init__()
        self.linear  = nn.Linear(config.patch_len, config.hidden_size)
        self.pos_enc = PositionalEncoding(
            pe_type = config.pe_type,
            hidden_size = config.hidden_size,
            learn_pe = config.learn_pe,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x) -> torch.Tensor:
        # x: [B, C, P, p]
        x = self.linear(x) # x: [B, C, P, d]
        x += self.pos_enc(x) # x: [B, C, P, p]
        x = self.dropout(x) # x: [B, C, P, p]

        return x
