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
    def __init__(self, mcfg):
        super().__init__()
        context_length = mcfg.context_length
        h = mcfg.h
        n_channels = mcfg.n_channels
        n_hist = getattr(mcfg, "n_hist", 1)

        in_features = context_length * n_channels * (1 + n_hist)
        out_features = h * n_channels
        self.fc = nn.Linear(in_features, out_features)
        self._ctx = context_length
        self.h = h
        self.n_channels = n_channels

    def forward(self, batch):
        # insample_y : [B, enc_size, C, 1+Vh]
        # outsample_y: [B, n_fcds, H, C]
        x      = batch["insample_y"]
        B      = x.shape[0]
        n_fcds = batch["outsample_y"].shape[1]

        x_ctx = x[:, -self._ctx:, :, :]                              # [B, L, C, 1+Vh]
        x_rep = x_ctx.unsqueeze(1).expand(B, n_fcds, -1, -1, -1)    # [B, n_fcds, L, C, 1+Vh]
        flat  = x_rep.reshape(B * n_fcds, -1)                        # [B*n_fcds, L*C*(1+Vh)]

        expected = self.fc.in_features
        if flat.shape[1] < expected:
            flat = torch.nn.functional.pad(flat, (0, expected - flat.shape[1]))
        else:
            flat = flat[:, :expected]

        out = self.fc(flat)                                           # [B*n_fcds, H*C]
        return out.reshape(B, n_fcds, self.h, self.n_channels)
