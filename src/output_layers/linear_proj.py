import torch
import torch.nn as nn

class Linear_Multivariate_Layer(nn.Module):
    """
    Flatten_Head
    """
    def __init__(self, config):
        super().__init__()

        self.multivariate_head = config.multivariate_head
        self.c_out = config.c_out

        if self.multivariate_head:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(config.nf, config.horizon * config.c_out))
                self.dropouts.append(nn.Dropout(config.head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(config.nf, config.horizon * config.c_out)
            self.dropout = nn.Dropout(config.head_dropout)

    def forward(self, x):  # x: [bs x n_channels x hidden_size x patch_num]
        n_channels = x.shape[1]
        if self.multivariate_head:
            x_out = []
            for i in range(n_channels):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x hidden_size * patch_num]
                z = self.linears[i](z)  # z: [bs x h]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x h]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x