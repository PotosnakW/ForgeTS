import torch.nn as nn

class Patching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_len = config.patch_len
        self.stride = config.stride

    def forward(self, x):
        x = x.unfold(dimension=-1, 
                     size=self.patch_len, 
                     step=self.stride)
        # x : [batch_size x n_channels x num_patch x patch_len]
        return x 