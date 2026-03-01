from typing import Optional
import torch
import math

class Masking:
    def __init__(self, 
                 mask_ratio : float = 0.3,
                 patch_len : int = 8,
                 stride : Optional[int] = None):
        """
        Indices with 0 mask are hidden, and with 1 are observed.
        """
        self.mask_ratio = mask_ratio    
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
    
    @staticmethod
    def convert_seq_to_patch_view(mask: torch.Tensor,
                                 patch_len: int = 8,
                                 stride: Optional[int] = None):
        """
        Input:
            mask : torch.Tensor of shape [batch_size x seq_len] or [batch_size x channels x seq_len]
            patch_len : int, length of each patch
            stride : int, step size between patches
        Output:
            mask : torch.Tensor of shape [batch_size x n_patches] or [batch_size x channels x n_patches]
        """
        stride = patch_len if stride is None else stride

        if mask.ndim == 3:
            batch_size, n_channels, seq_len = mask.shape
            mask = mask.unfold(dimension=-1, size=patch_len, step=stride) # [batch_size x channels x n_patches x patch_len]
            return (mask.sum(dim=-1) == patch_len).long()
        elif mask.ndim == 2:
            mask = mask.unfold(dimension=-1, size=patch_len, step=stride) # [batch_size x n_patches x patch_len]
            return (mask.sum(dim=-1) == patch_len).long()
    
    @staticmethod
    def convert_patch_to_seq_view(mask : torch.Tensor,
                                  patch_len : int = 8,):
        """
        Input:
            mask : torch.Tensor of shape [batch_size x n_patches]
        Output:
            mask : torch.Tensor of shape [batch_size x n_channels x seq_len]
        """
        return mask.repeat_interleave(patch_len, dim=-1)

    def generate_mask(self, x: torch.Tensor, input_mask: Optional[torch.Tensor] = None):
        """
        Input:
            x : torch.Tensor of shape [batch_size x n_channels x seq_len]
            input_mask: torch.Tensor of shape [batch_size x n_channels x n_patches]
        Output:
            mask : torch.Tensor of shape [batch_size x n_channels x seq_len]
        """

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        mask = self._mask_patch_view(x, input_mask=input_mask)
        mask = self.convert_patch_to_seq_view(mask, self.patch_len).long()
     
        return mask

    def _mask_patch_view(self, x, input_mask=None):
        """
        Input:
            x : torch.Tensor of shape [batch_size x n_channels x n_patches x patch_len]
            input_mask: torch.Tensor of shape [batch_size x n_channels x seq_len]
        Output:
            mask : torch.Tensor of shape [batch_size x n_patches]
        """
        batch_size, n_channels, n_patches, _ = x.shape
        
        input_mask = self.convert_seq_to_patch_view(
            input_mask, self.patch_len, self.stride
        )
        n_observed_patches = input_mask.sum(dim=-1, keepdim=True)

        len_keep = torch.ceil(n_observed_patches * (1 - self.mask_ratio)).long()
        noise = torch.rand(
            batch_size, n_channels, n_patches, device=x.device
        )  # noise in [0, 1], batch_size x n_channels x n_patches
        noise = torch.where(
            input_mask == 1, noise, torch.ones_like(noise)
        )  # only keep the noise of observed patches

        # Sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=-1
        )  # Ascend: small is keep, large is remove
        ids_restore = torch.argsort(
            ids_shuffle, dim=-1
        )  # ids_restore: [batch_size x n_channels x n_patches]

        # Create mask with 1 = keep, 0 = remove
        mask = torch.zeros([batch_size, n_channels, n_patches], device=x.device)
        for i in range(batch_size):
            for j in range(n_channels):
                mask[i, j, :len_keep[i, 0]] = 1  # Use len_keep[i, 0] to get the correct length for each batch

        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=-1, index=ids_restore) 

        return mask.long()
