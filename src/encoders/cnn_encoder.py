import torch
import torch.nn as nn


def _get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'gelu':
        return nn.GELU()
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
                 stride, groups, dilation, activation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        chomp = Chomp1d(padding)

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                groups=groups,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            chomp,
            nn.BatchNorm1d(out_channels),
            _get_activation(activation),
        )

    def forward(self, x):
        return self.block(x)


class CNNEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dims = [config.in_channels] + [config.hidden_size] * len(config.dilations)
        layers = []
        for i, dilation in enumerate(config.dilations):
            layers.append(
                ConvBlock(
                    in_channels=dims[i],
                    out_channels=dims[i + 1],
                    kernel_size=config.kernel_size,
                    groups=1,
                    stride=config.stride,
                    dilation=dilation,
                    activation=config.activation,
                )
            )
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, n_channels: int, **kwargs):
        # x: [B*C, 1, T]  ← Model.forward already reshapes
        x = self.encoder(x)    # [B*C, hidden_size, T]  ← was wrong: said [B, C, T]
        x = x.transpose(1, 2)  # [B*C, T, hidden_size]
        return x
