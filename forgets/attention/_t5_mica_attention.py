import torch
import torch.nn as nn
from typing import Optional, Union
import math

from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.models.t5.modeling_t5 import T5LayerNorm, T5Config

from ..common._modules import MLP


class T5Attention(nn.Module): # Default T5Attention copied from HuggingFace for version control
    def __init__(
        self,
        config: T5Config,
        has_relative_attention_bias=False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.layer_idx = layer_idx
        if layer_idx is None and self.is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.gradient_checkpointing = False

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None, cache_position=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        if cache_position is None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        else:
            context_position = cache_position[:, None].to(device)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        n_channels,
        hidden_states,
        attention_mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_values=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # attention_mask is (batch_size, 1, seq_length, seq_length)
        batch_size, seq_length = hidden_states.shape[:2]

        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        # Check is encoder-decoder model is being used. Otherwise we'll get `DynamicCache`
        is_updated = False
        if isinstance(past_key_values, EncoderDecoderCache):
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_values = past_key_values.cross_attention_cache
            else:
                curr_past_key_values = past_key_values.self_attention_cache
        else:
            curr_past_key_values = past_key_values

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_values.layers[self.layer_idx].keys
            value_states = curr_past_key_values.layers[self.layer_idx].values
        else:
            key_states = self.k(current_states)
            value_states = self.v(current_states)
            key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

            if past_key_values is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_values.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True

        if position_bias is None:
            key_length = key_states.shape[-2]
            # cache position is 0-indexed so we add 1 to get the real length of queries (aka with past)
            real_seq_length = query_length if query_length is not None else cache_position[-1] + 1
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length), device=hidden_states.device, dtype=hidden_states.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=hidden_states.device, cache_position=cache_position
                )
                position_bias = position_bias[:, :, -seq_length:, :]

            if attention_mask is not None: # [B*C, 1, P, P]
                attention_mask = (1.0 - attention_mask.float()) * -1e9
                position_bias = position_bias + attention_mask

        position_bias_masked = position_bias

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(-1, -2)) # same thing: torch.matmul(query_states, key_states.transpose(3, 2))
        # Below scaling not included in the T5 version but we added it to make implementation closer to original attention by Vaswani et al.
        scores = scores / math.sqrt(self.key_value_proj_dim)# [batch_size, n_heads, n_patch, n_patch]
        scores += position_bias_masked

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs
    
class T5MICAAttention(T5Attention):
    def __init__(self,
        config: T5Config,
        has_relative_attention_bias=False,
        layer_idx: Optional[int] = None,
        beta: Optional[torch.tensor] = None,
    ):
        super().__init__(config, has_relative_attention_bias, layer_idx)
        
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
            self.channel_attn = nn.Linear(config.d_kv, 1)
            self._compute_channel_weights = self._compute_query_channel_weights
        else:
            raise ValueError(f"mica_channel_weight_type '{config.mica_channel_weight_type}' not recognized. "
                    f"Use 'uniform', 'static', or 'dynamic'.")

        if config.mica_mixer_type.lower() == 'betas':
            self.mixing_gate = self.beta_mixing_gate
            if beta is not None:
                self.beta = beta
            else:
                if config.channelwise_beta:
                    self.beta = nn.Parameter(torch.rand((1, config.n_channels, config.num_heads, 1, 1))*1e-2)
                else:
                    self.beta = nn.Parameter(torch.rand((1, 1, config.num_heads, 1, 1))*1e-2)
                # Center values around 0
                with torch.no_grad():
                    self.beta -= self.beta.mean(dim=2, keepdim=True)
    
        elif config.mica_mixer_type.lower() == 'mlp':
            self.mixing_gate = self.mlp_mixing_gate
            self.mlp = MLP(
                in_features=config.d_kv * 2,
                out_features=config.d_kv,
                activation='ReLU',
                hidden_size=config.mlpmixer_hidden_size,
                num_layers=config.mlpmixer_n_layers,
                dropout=config.mlpmixer_dropout,
            )
        elif config.mica_mixer_type.lower() == 'mlp_query':
            self.mixing_gate = self.mlp_query_mixing_gate
            self.mlp = MLP(
                in_features=config.d_kv * 3,
                out_features=config.d_kv,
                activation='ReLU',
                hidden_size=config.mlpmixer_hidden_size,
                num_layers=config.mlpmixer_n_layers,
                dropout=config.mlpmixer_dropout,
            )
        else:
            raise ValueError(f"mica_mixer_type '{config.mica_mixer_type}' not recognized. "
                    f"Use 'betas', 'mlp', 'mlp_query', or 'none'.")

    def compute_bias(self, query_length, key_length, device=None, cache_position=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        if cache_position is None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        else:
            context_position = cache_position[:, None].to(device)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0).unsqueeze(0)  # shape (1, 1, num_heads, query_length, key_length) --> NEW: added dimension=1 for n_channels
        return values
    
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
        n_channels,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_values=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, 1, 1, key_length) (non-causal encoder) or (batch_size, 1, n_patch, key_length) (causal decoder)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, 
                                         -1, 
                                         self.n_heads, 
                                         self.key_value_proj_dim).transpose(1, 2)  # [batch_size, n_heads, n_patch, dim]
        query_states = query_states.view(batch_size//n_channels, 
                                         n_channels, 
                                         self.n_heads, 
                                         seq_length,
                                         self.key_value_proj_dim) # [batch_size, n_channels, n_heads, n_patch, dim]

        # Check is encoder-decoder model is being used. Otherwise we'll get `DynamicCache`
        is_updated = False
        if isinstance(past_key_values, EncoderDecoderCache):
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_values = past_key_values.cross_attention_cache
            else:
                curr_past_key_values = past_key_values.self_attention_cache
        else:
            curr_past_key_values = past_key_values

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_values.layers[self.layer_idx].keys
            value_states = curr_past_key_values.layers[self.layer_idx].values
        else:
            key_states = self.k(current_states)
            value_states = self.v(current_states)
            key_states = key_states.view(batch_size, 
                                         -1, 
                                         self.n_heads, 
                                         self.key_value_proj_dim).transpose(1, 2)
            key_states = key_states.view(batch_size//n_channels, 
                                         n_channels, 
                                         self.n_heads, 
                                         seq_length,
                                         self.key_value_proj_dim) # [batch_size, n_channels, n_heads, n_patch, dim]
            value_states = value_states.view(batch_size, 
                                             -1, 
                                             self.n_heads, 
                                             self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size//n_channels, 
                                             n_channels, 
                                             self.n_heads, 
                                             seq_length, 
                                             self.key_value_proj_dim) # [batch_size, n_channels, n_heads, n_patch, dim]

            if past_key_values is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_values.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True

        if position_bias is None:
            key_length = key_states.shape[-2]
            # cache position is 0-indexed so we add 1 to get the real length of queries (aka with past)
            real_seq_length = query_length if query_length is not None else cache_position[-1] + 1
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, 1, self.n_heads, seq_length, key_length), device=hidden_states.device, dtype=hidden_states.dtype
                ) # NEW: added dim(1) for n_channels
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=hidden_states.device, cache_position=cache_position
                )
                position_bias = position_bias[:, :, :, -seq_length:, :]

            if mask is not None:
                #causal_mask = mask[:, :, :, :, : key_states.shape[-2]]
                #position_bias = position_bias + causal_mask
                mask = mask.view(batch_size//n_channels, n_channels, 1, 1, key_states.shape[-2])
                mask = (1.0 - mask.float()) * -1e9
                position_bias = position_bias + mask

        position_bias_masked = position_bias

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(-1, -2)) # [batch_size, n_channels, n_heads, n_patch, n_patch]
        # Below scaling not included in the T5 version but we added it to make implementation closer to original attention by Vaswani et al.
        scores = scores / math.sqrt(self.key_value_proj_dim)# [batch_size, n_channels, n_heads, n_patch, n_patch]
        scores += position_bias_masked # [batch_size, n_channels, n_heads, n_patch, n_patch]
    
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores) # [batch_size, n_channels, n_heads, n_patch, n_patch]
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training) # [batch_size, n_channels, n_heads, n_patch, n_patch]

        attn_output = torch.matmul(attn_weights, value_states) # [batch_size, n_channels, n_heads, n_patch, dim]

        # Infini attention computation across channels
        memory_matrix, z = self._update_memory_matrix(
            key_states=key_states, 
            value_states=value_states, 
            query_states=query_states, 
            n_channels=n_channels,
        )
        A_mem = self.retrieve_from_memory(
            query_states=query_states,
            memory_matrix=memory_matrix,
            z=z,
        )

        # Channel mixing
        attn_output = self.mixing_gate(
            a_mem=A_mem, 
            attn_output=attn_output, 
            query_states=query_states,
        ) # [batch_size, n_channels, n_heads, n_patch, dim]
        
        attn_output = attn_output.transpose(2, 3).contiguous() # [batch_size, n_channels, n_patch, n_heads, dim]
        attn_output = attn_output.view(batch_size, -1, self.inner_dim) # [batch_size*n_channels, n_patch, n_heads*dim]
        attn_output = self.o(attn_output) # [batch_size*n_channels, n_patch, n_heads*dim]

        outputs = (attn_output, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

class T5LayerSelfAttention(nn.Module):
    def __init__(self, 
                 config, 
                 has_relative_attention_bias=False, 
                 layer_idx: Optional[int] = None, 
                 beta: Optional[torch.tensor] = None, 
        ):
        super().__init__()

        if config.mica_mixer_type.lower() in ['betas', 'mlp', 'mlp_query']:
            self.SelfAttention = T5MICAAttention(
                config=config, 
                has_relative_attention_bias=has_relative_attention_bias, 
                layer_idx=layer_idx, 
                beta=beta,
            )
        elif config.mica_mixer_type.lower() == 'none':
            self.SelfAttention = T5Attention(
                config=config,
                has_relative_attention_bias=has_relative_attention_bias, 
                layer_idx=layer_idx
            )
        else:
            raise ValueError(f"Channel mixing method: {config.mica_mixer_type} not recognized. "
                            f"Use 'betas', 'mlp', 'mlp_query', or 'none'.")

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        n_channels,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            n_channels=n_channels,
            hidden_states=normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them

        return outputs
    
class T5LayerCrossAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None, beta: Optional[torch.tensor] = None):
        super().__init__()

        if config.mica_mixer_type.lower() in ['betas', 'mlp', 'mlp_query']:
            self.EncDecAttention = T5MICAAttention(
                config=config, 
                has_relative_attention_bias=False, 
                layer_idx=layer_idx, 
                beta=beta,
            )
        elif config.mica_mixer_type.lower() == 'none':
            self.EncDecAttention = T5Attention(
                config=config, 
                has_relative_attention_bias=False, 
                layer_idx=layer_idx
            )
        else:
            raise ValueError(f"Channel mixing method: {config.mica_mixer_type} not recognized. "
                            f"Use 'betas', 'mlp', 'mlp_query', or 'none'.")
    
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(
        self,
        n_channels,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        past_key_values=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        cache_position=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            n_channels=n_channels,
            hidden_states=normed_hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs