import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    nn.LSTM with batch_first=True handles the sequential computation internally.
    Given input [B*C, seq_len, hidden_size], it processes all timesteps in order
    and returns [B*C, seq_len, hidden_size] where each position's output
    encodes all prior context via the recurrent hidden state.
    """
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = config.hidden_size,
            hidden_size = config.hidden_size,
            num_layers = config.n_layers,
            batch_first = True,
            dropout = config.dropout if config.n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        inputs_embeds:  torch.Tensor,
        **kwargs,
    ):

        out, _ = self.lstm(inputs_embeds)   # [B*C, S, hidden_size]
        out = self.dropout(out)

        return out