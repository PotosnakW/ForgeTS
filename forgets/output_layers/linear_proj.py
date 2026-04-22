import torch
import torch.nn as nn


class LinearProjectionLayer(nn.Module):
    """
    Shared linear projection across all channels.
    Flattens [B, C, P, d_model] → projects to [B, C, H*c_out].
    Efficient and parameter-efficient — same weights for all channels.
    """
    def __init__(self, config):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear  = nn.Linear(config.nf, config.horizon * config.c_out)
        self.dropout = nn.Dropout(config.head_dropout)

    def forward(self, x) -> torch.Tensor:
        # x: [B, C, T, P, d_model]
        x = self.flatten(x)   # [B, C, T, P*d_model]
        x = self.linear(x)    # [B, C, T, H*c_out]
        x = self.dropout(x)
        return x


class LinearProjectionLayerMultivariate(nn.Module):
    """
    Per-channel linear projection.
    Each channel has its own linear layer and dropout — more expressive
    but scales linearly with number of channels.
    Requires fixed number of channels at init time.
    """
    def __init__(self, config):
        super().__init__()
        n_vars = config.n_vars
        self.flattens = nn.ModuleList([nn.Flatten(start_dim=-2) for _ in range(n_vars)])
        self.linears  = nn.ModuleList([nn.Linear(config.nf, config.horizon * config.c_out) for _ in range(n_vars)])
        self.dropouts = nn.ModuleList([nn.Dropout(config.head_dropout) for _ in range(n_vars)])

    def forward(self, x) -> torch.Tensor:
        # x: [B, C, T, P, d_model]
        x_out = []
        for i in range(x.shape[1]):
            z = self.flattens[i](x[:, i])   # [B, T, P*d_model]
            z = self.linears[i](z)           # [B, T, H*c_out]
            z = self.dropouts[i](z)
            x_out.append(z)
        return torch.stack(x_out, dim=1)    # [B, C, T, H*c_out]