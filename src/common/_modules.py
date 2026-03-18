import warnings
import torch
from torch import nn
import math


class MLP(nn.Module):
    """Multi-Layer Perceptron Class

    **Parameters:**<br>
    `in_features`: int, dimension of input.<br>
    `out_features`: int, dimension of output.<br>
    `activation`: str, activation function to use.<br>
    `hidden_size`: int, dimension of hidden layers.<br>
    `num_layers`: int, number of hidden layers.<br>
    `dropout`: float, dropout rate.<br>
    """

    def __init__(
        self, in_features, out_features, activation, hidden_size, num_layers, dropout
    ):
        super().__init__()
        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"

        self.activation = getattr(nn, activation)()

        # MultiLayer Perceptron
        # Input layer
        layers = [
            nn.Linear(in_features=in_features, out_features=hidden_size),
            self.activation,
            nn.Dropout(dropout),
        ]
        # Hidden layers
        for i in range(num_layers - 2):
            layers += [
                nn.Linear(in_features=hidden_size, out_features=hidden_size),
                self.activation,
                nn.Dropout(dropout),
            ]
        # Output layer
        layers += [nn.Linear(in_features=hidden_size, out_features=out_features)]

        # Store in layers as ModuleList
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class RevIN(nn.Module):
    """RevIN (Reversible-Instance-Normalization)"""

    def __init__(
        self,
        eps=1e-5,
        affine=False,
        subtract_last=False,
        non_norm=False,
    ):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        :param substract_last: if True, the substraction is based on the last value
                               instead of the mean in normalization
        :param non_norm: if True, no normalization performed.
        """
        super(RevIN, self).__init__()
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class PositionalEncoding(nn.Module):
    """
    Unified positional encoding module supporting:
      - Sin/Cos encodings (1D or 2D)
      - Linear / Exponential coordinate encodings (1D or 2D)
      - Random (normal/uniform/zeros) encodings
      - Learnable or fixed encodings
    """

    def __init__(
        self,
        pe_type: str = "sincos",   # ['sincos', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'gauss', 'uniform', 'zeros', 'zero', None]
        q_len: int = 5000,
        hidden_size: int = 768,
        learn_pe: bool = False,
        normalize: bool = True,
    ):
        super().__init__()

        self.pe_type = pe_type
        self.q_len = q_len
        self.hidden_size = hidden_size
        self.learn_pe = learn_pe
        self.normalize = normalize

        # Build encoding tensor
        W_pos = self._build_encoding()
        if self.learn_pe:
            self.W_pos = nn.Parameter(W_pos)
        else:
            self.register_buffer("W_pos", W_pos)

    def _build_encoding(self):
        pe = self.pe_type

        if pe is None:
            W_pos = torch.empty((self.q_len, self.hidden_size))
            nn.init.uniform_(W_pos, -0.02, 0.02)

        elif pe in ["zero", "zeros"]:
            W_pos = torch.zeros((self.q_len, self.hidden_size))

        elif pe in ["gauss", "normal"]:
            W_pos = torch.empty((self.q_len, self.hidden_size))
            nn.init.normal_(W_pos, mean=0.0, std=0.1)

        elif pe == "uniform":
            W_pos = torch.empty((self.q_len, self.hidden_size))
            nn.init.uniform_(W_pos, a=0.0, b=0.1)

        elif pe == "sincos":
            W_pos = self._sin_cos_encoding(self.q_len, self.hidden_size)

        elif pe in ["lin1d", "exp1d"]:
            W_pos = self._coord1d_encoding(self.q_len, exponential=("exp" in pe))

        elif pe in ["lin2d", "exp2d"]:
            W_pos = self._coord2d_encoding(self.q_len, self.hidden_size, exponential=("exp" in pe))

        else:
            raise ValueError(
                f"{pe} is not a valid positional encoding type. "
                f"Available: None, 'zeros', 'zero', 'normal', 'uniform', 'lin1d', "
                f"'exp1d', 'lin2d', 'exp2d', 'sincos'."
            )

        if self.normalize:
            W_pos = (W_pos - W_pos.mean()) / (W_pos.std() * 10)

        return W_pos

    def _sin_cos_encoding(self, q_len, hidden_size):
        """Classic sinusoidal encoding"""
        pe = torch.zeros(q_len, hidden_size)
        position = torch.arange(0, q_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _coord1d_encoding(self, q_len, exponential=False):
        """1D coordinate encoding (linear or exponential)"""
        exponent = 0.5 if exponential else 1.0
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** exponent) - 1
        return cpe

    def _coord2d_encoding(self, q_len, hidden_size, exponential=False, eps=1e-3):
        """2D coordinate encoding (linear or exponential)"""
        x = 0.5 if exponential else 1
        for _ in range(100):
            cpe = (
                2
                * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x)
                * (torch.linspace(0, 1, hidden_size).reshape(1, -1) ** x)
                - 1
            )
            mean = cpe.mean()
            if abs(mean) <= eps:
                break
            x += -0.001 if mean > eps else 0.001
        return cpe

    def forward(self, x):
        """
        Returns positional encodings broadcastable to input tensor `x`.
        Accepts:
          - [batch_size, n_channels, seq_len, d_model]
        """
        seq_len = x.size(2)
        pe = self.W_pos[:seq_len]
        return pe.unsqueeze(0).unsqueeze(0) #[batch_size, n_channels, seq_len, d_model]

class Patching(nn.Module):
    def __init__(self, 
                 patch_len : int, 
                 stride : int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        x = x.unfold(dimension=-1, 
                     size=self.patch_len, 
                     step=self.stride)
        # x : [batch_size x n_channels x num_patch x patch_len]
        return x 
    
class Transpose(nn.Module):
    """
    Transpose
    """

    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)

def _make_causal_token_mask(
    key_padding_mask: torch.Tensor,  # [B, C, L] — 1/True = VALID, 0/False = INVALID
    device: torch.device,
) -> torch.Tensor:
    B, C, L = key_padding_mask.shape

    # Causal mask: [1, 1, 1, L, L] — 1 where attention is ALLOWED
    causal_mask = torch.ones(L, L, dtype=torch.float, device=device).tril()
    causal_mask = causal_mask.view(1, 1, 1, L, L)

    # Token validity: [B, C, 1, 1, L] — 1 = valid key, 0 = invalid key
    token_mask = key_padding_mask.float().unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1, L]

    # Combine: 0 if EITHER is blocked
    combined = causal_mask * token_mask  # [B, C, 1, L, L]

    return combined  # [B, C, 1, L, L] — 1=attend, 0=block