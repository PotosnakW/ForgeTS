import torch
import torch.nn as nn



def _get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'gelu':
        return nn.GELU()                    # no inplace for GELU
    else:
        raise ValueError(f"Activation '{activation}' not recognized. Use 'relu' or 'gelu'.")


class Chomp1d(nn.Module):
    """Remove trailing time steps introduced by causal padding."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride, dilation, activation, causal=True):
        super().__init__()                  # no args to super().__init__()
        
        if causal:
            padding = (kernel_size - 1) * dilation  # left-pad only → no future leakage
            chomp = Chomp1d(padding)
        else:
            padding = (kernel_size - 1) * dilation // 2  # symmetric → same length out
            chomp = nn.Identity()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            chomp,
            nn.BatchNorm1d(out_channels),   # no trailing comma (was making a tuple)
            _get_activation(activation),
        )

    def forward(self, x):
        return self.block(x)


class CNNEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # build channel dims: in_channels → hidden → ... → hidden
        dims = [config.in_channels] + [config.hidden_size] * len(config.dilations)

        layers = []
        for i, dilation in enumerate(config.dilations):
            layers.append(
                ConvBlock(
                    in_channels=dims[i],
                    out_channels=dims[i + 1],
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    dilation=dilation,
                    activation=config.activation,
                    causal=config.causal,
                )
            )

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, n_channels: int, **kwargs):

        B, C, T = x.shape # [B, C, T]
        x = self.encoder(x)   # [B, C, T]
        x = x.transpose(1, 2)  # [B*C, seq_len, hidden_size]
        return x