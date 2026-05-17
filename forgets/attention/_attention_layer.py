import torch
import torch.nn as nn
from typing import Optional


from ._sdp_attention import ScaledDotProductAttention
from ._mica_attention import MICAScaledDotProductAttention


class MultiheadAttention(nn.Module):
    """
    Multi-Head Attention with optional MICA.
    Traditional format similar to standard Transformer implementations.
    """
    
    def __init__(
        self,
        config,
        beta: Optional[torch.tensor] = None,
    ):

        super().__init__()
        assert (
            not config.hidden_size % config.n_heads
        ), f"hidden_size ({config.hidden_size}) must be divisible by n_heads ({config.n_heads})"
        self.d_k = config.hidden_size // config.n_heads if config.d_k is None else config.d_k
        self.d_v = config.hidden_size // config.n_heads if config.d_v is None else config.d_v

        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.mica_mixer_type = config.mica_mixer_type.lower()
        self.res_attention = config.res_attention
        
        # Q, K, V projections
        self.W_Q = nn.Linear(config.hidden_size, self.d_k * config.n_heads, bias=config.qkv_bias)
        self.W_K = nn.Linear(config.hidden_size, self.d_k * config.n_heads, bias=config.qkv_bias)
        self.W_V = nn.Linear(config.hidden_size, self.d_v * config.n_heads, bias=config.qkv_bias)
        
        # Scaled Dot-Product Attention (vanilla or mica)
        if config.mica_mixer_type.lower() in ['betas', 'mlp', 'mlp_query']:
            self.sdp_attn = MICAScaledDotProductAttention(
                config=config,
                d_k=self.d_k,
                d_v=self.d_v,
                beta=beta,
            )
        elif config.mica_mixer_type == 'none':
            self.sdp_attn = ScaledDotProductAttention(
                config=config,
                d_k=self.d_k,
                d_v=self.d_v,
            )
        else:
            raise ValueError(f"Channel mixing method: {config.mica_mixer_type} not recognized. "
                            f"Use 'betas', 'mlp', 'mlp_query', or 'none'.")

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(config.n_heads * self.d_v, config.hidden_size),
            nn.Dropout(config.proj_dropout)
        )
    
    def forward(
        self,
        n_channels: int,
        Q: torch.Tensor,
        attention_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
        channel_mask: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        prev: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for multi-head attention.
        
        Input shape (without channels):
            Q: [bs x seq_len x hidden_size]
            K: [bs x seq_len x hidden_size] (optional, defaults to Q)
            V: [bs x seq_len x hidden_size] (optional, defaults to Q)
            
        Input shape (with channels):
            Q: [bs*n_channels x seq_len x hidden_size]
            (internally reshaped to [bs x n_channels x seq_len x hidden_size])
            
        Output shape:
            output: [bs*n_channels x seq_len x hidden_size]
            A_mem: [bs x n_channels x n_heads x seq_len x d_v] (if mica_mixer_type != 'none')
            attn_weights: [bs*n_channels x 1 x seq_len x seq_len]
        """

        batch_size = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q
        
        # Linear projections and split into multiple heads
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k)  # [bs x seq_len x n_heads x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k)  # [bs x seq_len x n_heads x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v)  # [bs x seq_len x n_heads x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q=q_s,
                k=k_s,
                v=v_s,
                n_channels=n_channels,
                prev=prev,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                channel_mask=channel_mask,
            )
        else:
            output, attn_weights = self.sdp_attn(
                q=q_s, 
                k=k_s, 
                v=v_s, 
                n_channels=n_channels,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                channel_mask=channel_mask,
            )
        
        # Final output projection
        output = self.to_out(output)
        
        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights
    