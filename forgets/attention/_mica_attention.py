import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from ..common._modules import MLP

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
        prev: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Scaled Dot-Product Attention.
        
        Input shape:
            q: [bs * n_channels x seq_len x n_heads x d_k]
            k: [bs * n_channels x seq_len x n_heads x d_k]
            v: [bs * n_channels x seq_len x n_heads x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            attention_mask       : [1 x seq_len x seq_len]
            
        Output shape:
            output: [bs x n_heads x seq_len x d_v]
            attn_weights: [bs x n_heads x seq_len x seq_len]
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
        
        if attention_mask is not None:
            # Both bool and float use same convention: 1/True=attend, 0/False=block
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

class MICAScaledDotProductAttention(ScaledDotProductAttention):
    """
    Scaled Dot-Product Attention with MICA.
    Based on "Attention is All You Need" (Vaswani et al., 2017) and 
    "Leave No Context Behind" (Munkhdalai et al., 2024).
    """
    
    def __init__(
        self,
        config,
        d_k,
        d_v,
        beta: Optional[torch.tensor] = None,
    ):
        super().__init__(config=config, d_k=d_k, d_v=d_v)

        self.elu = nn.ELU()
        
        # Select memory update/retrieval methods based on channel exclusion
        if config.mica_channel_exclusion:
            self._update_memory_matrix = self._update_memory_matrix_channelexl
        else:
            self._update_memory_matrix = self._update_memory_matrix_allchannels

        # Select channel weight type:
        if config.mica_channel_weight_type == 'uniform':
            self._compute_channel_weights = self._compute_uniform_channel_weights
        elif config.mica_channel_weight_type == 'static':
            self.channel_weights = nn.Parameter(torch.ones(1, config.n_channels, config.n_heads, 1, 1))
            self._compute_channel_weights = self._compute_static_channel_weights
        elif config.mica_channel_weight_type == 'dynamic':
            self.channel_attn = nn.Linear(d_k, 1)
            self._compute_channel_weights = self._compute_query_channel_weights
        else:
            raise ValueError(f"mica_channel_weight_type '{config.mica_channel_weight_type}' not recognized. "
                    f"Use 'uniform', 'static', or 'dynamic'.")

        # Select gate mechanism
        if config.mica_mixer_type.lower() == 'betas':
            self.mixing_gate = self.beta_mixing_gate
            if beta is not None:
                self.beta = beta
            else:
                if config.channelwise_beta:
                    self.beta = nn.Parameter(torch.rand((1, config.n_channels, config.n_heads, 1, 1))*1e-2)
                else:
                    self.beta = nn.Parameter(torch.rand((1, 1, config.n_heads, 1, 1))*1e-2)
                # Center values around 0
                with torch.no_grad():
                    self.beta -= self.beta.mean(dim=2, keepdim=True)
    
        elif config.mica_mixer_type.lower() == 'mlp':
            self.mixing_gate = self.mlp_mixing_gate
            self.mlp = MLP(
                in_features=d_v * 2,
                out_features=d_v,
                activation='ReLU',
                hidden_size=config.mlpmixer_hidden_size,
                num_layers=config.mlpmixer_n_layers,
                dropout=config.mlpmixer_dropout,
            )
        elif config.mica_mixer_type.lower() == 'mlp_query':
            self.mixing_gate = self.mlp_query_mixing_gate
            self.mlp = MLP(
                in_features=d_v * 3,
                out_features=d_v,
                activation='ReLU',
                hidden_size=config.mlpmixer_hidden_size,
                num_layers=config.mlpmixer_n_layers,
                dropout=config.mlpmixer_dropout,
            )
        else:
            raise ValueError(f"mica_mixer_type '{config.mica_mixer_type}' not recognized. "
                    f"Use 'betas', 'mlp', 'mlp_query', or 'none'.")
    
    def _compute_uniform_channel_weights(self, query_states):
        return torch.ones(1, 1, 1, 1, 1, device=query_states.device)
    
    def _compute_static_channel_weights(self, query_states):
        return torch.softmax(self.channel_weights, dim=1)  # [1, C, H, 1, 1]
    
    def _compute_query_channel_weights(self, query_states):
        # query_states: [B, C, H, P, D]
        q_pooled = query_states.mean(dim=3)   # [B, C, H, D]
        scores = self.channel_attn(q_pooled)  # [B, C, H, 1]
        return torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, C, H, 1, 1]

    def _update_memory_matrix_allchannels(self, key_states, value_states, query_states, n_channels):
        w = self._compute_channel_weights(query_states)
        sigma_k = self.elu(key_states) + 1.0  # [batch_size, n_channels, n_heads, n_patch, dim]
        sigma_k_T = sigma_k.transpose(-2, -1) # [batch_size, n_channels, n_heads, dim, n_patch]

        memory_matrix = torch.matmul(sigma_k_T, value_states) # [B, C, H, D, D]
        memory_matrix = (w * memory_matrix).sum(dim=1, keepdim=True) # [batch_size, 1, n_heads, dim, dim] sum over channels
        
        z = sigma_k.sum(dim=-2).unsqueeze(-1)
        z = (w * z).sum(dim=1, keepdim=True) # [batch_size, n_heads, dim, 1] sum over sequence length and channels
    
        return memory_matrix, z

    def _update_memory_matrix_channelexl(self, key_states, value_states, query_states, n_channels):
        w = self._compute_channel_weights(query_states)
        sigma_k = self.elu(key_states) + 1.0  # [batch_size, n_channels, n_heads, n_patch, dim]
        sigma_k_T = sigma_k.transpose(-2, -1) # [batch_size, n_channels, n_heads, dim, n_patch]

        C = key_states.shape[1]
        per_channel_mm = torch.matmul(sigma_k_T, value_states)  # [B, C, H, D, D]
        weighted_sum = (w * per_channel_mm).sum(dim=1, keepdim=True)  # [B, 1, H, D, D]
        memory_matrix = weighted_sum.expand(-1, C, -1, -1, -1) - w * per_channel_mm  # [B, C, H, D, D]

        per_channel_z = sigma_k.sum(dim=-2).unsqueeze(-1)  # [B, C, H, D, 1]
        weighted_z_sum = (w * per_channel_z).sum(dim=1, keepdim=True)  # [B, 1, H, D, 1]
        z = weighted_z_sum.expand(-1, C, -1, -1, -1) - w * per_channel_z  # [B, C, H, D, 1]

        return memory_matrix, z

    def retrieve_from_memory(self, query_states, memory_matrix, z):
        sigma_q = self.elu(query_states) + 1.0  # [B, C, H, P, D]
        numerator = sigma_q @ memory_matrix         # [B, C, H, P, D]
        denominator = (sigma_q @ z) + 1e-6 # [B, C, H, P, 1]
        A_mem = numerator / denominator             # [B, C, H, P, D]
    
        return A_mem
    
    def beta_mixing_gate(self, a_mem, attn_output, query_states):
        """Learned interpolation between memory and attention using per-head betas."""
        attn_output = torch.sigmoid(self.beta) * a_mem + (1 - torch.sigmoid(self.beta)) * attn_output 
        return attn_output
    
    def mlp_mixing_gate(self, a_mem, attn_output, query_states):
        """Context-aware mixing via MLP on concatenated memory and attention."""
        attn_output = torch.cat([a_mem, attn_output], dim=-1)  # [batch_size, n_channels, n_heads, n_patch, dim*2]
        attn_output = self.mlp(attn_output)  # [batch_size, n_channels, n_heads, n_patch, dim]
        return attn_output
    
    def mlp_query_mixing_gate(self, a_mem, attn_output, query_states):
        """Context-aware mixing via MLP with query, memory, and attention."""
        attn_output = torch.cat([a_mem, attn_output, query_states], dim=-1)  # [batch_size, n_channels, n_heads, n_patch, dim*3]
        attn_output = self.mlp(attn_output)  # [batch_size, n_channels, n_heads, n_patch, dim]
        return attn_output

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        n_channels: int,
        prev: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Scaled Dot-Product Attention with memory mechanism.
        
        Input shape (with channels):
            q: [bs * n_channels x seq_len x n_heads x d_k]
            k: [bs * n_channels x seq_len x n_heads x d_k]
            v: [bs * n_channels x seq_len x n_heads x d_v]
            n_channels: int
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attention_mask       : [1 x seq_len x seq_len]
            
        Output shape:
            output: [bs x n_channels x n_heads x seq_len x d_v]
            attn_weights: [bs x n_channels x n_heads x seq_len x seq_len]
        """

        batch_size = q.shape[0] // n_channels
        seq_len = q.shape[1]

        q = q.view(batch_size, n_channels, seq_len, self.n_heads, -1)
        q = q.transpose(2, 3).contiguous()  # [bs x n_channels x n_heads x seq_len x d_k]
            
        k = k.view(batch_size, n_channels, seq_len, self.n_heads, -1)
        k = k.permute(0, 1, 3, 4, 2).contiguous()  # [bs x n_channels x n_heads x d_k x seq_len]
            
        v = v.view(batch_size, n_channels, seq_len, self.n_heads, -1)
        v = v.transpose(2, 3).contiguous()  # [bs x n_channels x n_heads x seq_len x d_v]
        
        # Scaled MatMul (q, k) - compute attention scores
        attn_scores = torch.matmul(q, k) * self.scale  # Vaswani et al. scaling

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None:
            prev = prev.view(batch_size, n_channels, self.n_heads, seq_len, seq_len)
            attn_scores = attn_scores + prev
        
        if attention_mask is not None:
            # Both bool and float use same convention: 1/True=attend, 0/False=block
            attention_mask = attention_mask.reshape(batch_size, n_channels, 1, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Normalize attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute attention output (v is [B, C, H, P, D])
        output = torch.matmul(attn_weights, v)
        
        # Infini-attention: retrieve from memory
        # k_for_memory should be [B, C, H, P, D] (not transposed)
        k_for_memory = k.transpose(-2, -1)
        memory_matrix, z = self._update_memory_matrix(
            key_states=k_for_memory, 
            value_states=v, 
            query_states=q, 
            n_channels=n_channels,
        )
        A_mem = self.retrieve_from_memory(
            query_states=q,
            memory_matrix=memory_matrix,
            z=z,
        )

        # Channel mixing
        output = self.mixing_gate(
            a_mem=A_mem, 
            attn_output=output, 
            query_states=q,
        ) # [bs x n_channels x n_heads x seq_len x d_v]

        output = output.transpose(2, 3).contiguous()  # [bs x n_channels x seq_len x n_heads x d_v]
        output = output.view(batch_size*n_channels, -1, self.inner_dim)  # [bs*n_channels x seq_len x n_heads*d_v]
        attn_weights = attn_weights.view(batch_size*n_channels, self.n_heads, seq_len, seq_len)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights
        
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
        K: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        prev: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
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
            )
        else:
            output, attn_weights = self.sdp_attn(
                q=q_s, 
                k=k_s, 
                v=v_s, 
                n_channels=n_channels,
                attention_mask=attention_mask
            )
        
        # Final output projection
        output = self.to_out(output)
        
        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights
    