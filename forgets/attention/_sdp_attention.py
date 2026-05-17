import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """
    Vanilla Scaled Dot-Product Attention.
    Based on "Attention is All You Need" (Vaswani et al., 2017).
    """
    
    def __init__(
        self,
        config,
        d_k,
        d_v,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.scale = d_k ** -0.5
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.res_attention = config.res_attention
        self.inner_dim = config.n_heads * d_v
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        n_channels: int,
        attention_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
        channel_mask: torch.Tensor,
        prev: Optional[torch.Tensor] = None,
    ):
        """
        Scaled Dot-Product Attention.
        
        Input shape:
            q: [bs * n_channels x seq_len x n_heads x d_k]
            k: [bs * n_channels x seq_len x n_heads x d_k]
            v: [bs * n_channels x seq_len x n_heads x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            attention_mask: [bs*n_channels x 1 x seq_len x seq_len] — 1=attend, 0=block.
            channel_mask  : [bs x n_channels] — 1=real, 0=padded.
            
        Output shape:
            output: [bs*n_channels x seq_len x n_heads*d_v]
            attn_weights: [bs*n_channels x n_heads x seq_len x seq_len]
        """

        batch_size = q.shape[0]

        q = q.transpose(1, 2)  # [bs x n_heads x seq_len x d_k]
        k = k.permute(0, 2, 3, 1)  # [bs x n_heads x d_k x seq_len]
        v = v.transpose(1, 2)  # [bs x n_heads x seq_len x d_v]
        
        # Scaled MatMul (q, k) - compute attention scores
        attn_scores = torch.matmul(q, k) * self.scale  # Vaswani et al. scaling

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None:
            attn_scores = attn_scores + prev

        attention_mask = attention_mask * channel_mask.view(-1, 1, 1, 1).float()
        attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Normalize attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute attention output
        output = torch.matmul(attn_weights, v) # [bs x n_heads x seq_len x d_v]

        # Reshape back
        output = output.transpose(1, 2).contiguous()  # [bs x seq_len x n_heads x d_v]
        output = output.view(batch_size, -1, self.inner_dim)  # [bs x seq_len x n_heads*d_v]
        
        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights