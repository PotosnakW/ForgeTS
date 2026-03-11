from common._base_model import BaseModel
import torch.nn as nn
import torch

class TinyLinearModel(BaseModel):
    """
    Simplest possible BaseModel subclass for pipeline testing.

    forward: flatten insample_y → linear → reshape to [B*n_fcds, H, C]
    compute_loss: MSE vs outsample_y (inherited default).

    Not useful for forecasting — only here to exercise the pipeline end-to-end.
    """
    def __init__(self, context_length: int, horizon: int, n_channels: int,
                 n_hist: int = 1):
        super().__init__()
        in_features  = context_length * n_channels * (1 + n_hist)
        out_features = horizon * n_channels
        self.fc = nn.Linear(in_features, out_features)
        self._ctx = context_length
        self._H   = horizon
        self._C   = n_channels

    def forward(self, batch):
        # insample_y : [B, enc_size, C, 1+Vh]
        # outsample_y: [B, n_fcds, H, C]
        x       = batch["insample_y"]                # [B, enc_size, C, 1+Vh]
        B       = x.shape[0]
        n_fcds  = batch["outsample_y"].shape[1]

        # Take only the last context_length steps, flatten per FCD
        x_ctx   = x[:, -self._ctx:, :, :]           # [B, L, C, 1+Vh]
        # Repeat for each FCD (naive — just for shape correctness)
        x_rep   = x_ctx.unsqueeze(1).expand(B, n_fcds, -1, -1, -1)  # [B, n_fcds, L, C, 1+Vh]
        flat    = x_rep.reshape(B * n_fcds, -1)     # [B*n_fcds, L*C*(1+Vh)]

        # Pad/trim to expected in_features
        expected = self.fc.in_features
        if flat.shape[1] < expected:
            flat = torch.nn.functional.pad(flat, (0, expected - flat.shape[1]))
        else:
            flat = flat[:, :expected]

        out = self.fc(flat)                          # [B*n_fcds, H*C]
        return out.reshape(B, n_fcds, self._H, self._C)
