from dataclasses import dataclass
import torch

@dataclass
class VariableForecastSettings:
    max_input_length : int
    max_forecast_horizon : int
    patch_len : int
    d_model : int
    patch_stride_len : int

    def head_num_feats(self):
        num_patches = (
            (max(self.max_input_length, self.patch_len) - self.patch_len) 
            // self.patch_stride_len + 1
        )
        return num_patches * self.d_model
    
class MLPForecastHead(torch.nn.Module):
    def __init__(self, 
                variable_forecast_settings : VariableForecastSettings,
                intermediate_layers : list[int] = [512, 512],
                dropout = 0.1):

        super().__init__()

        self.flatten = torch.nn.Flatten(start_dim=-2)

        self.variable_forecast_settings = variable_forecast_settings

        input_dim = variable_forecast_settings.head_num_feats()
        output_dim = variable_forecast_settings.max_forecast_horizon

        layer_dims = [input_dim] + intermediate_layers

        self.layers = torch.nn.ModuleList()

        for input, output in zip(range(len(layer_dims) - 1), range(1, len(layer_dims))):
            self.layers.append(torch.nn.Linear(layer_dims[input], layer_dims[output]))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(dropout))

        self.layers.append(torch.nn.Linear(layer_dims[-1], output_dim))

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        return x

def variable_forecast_loss(forecast_pred, 
                           forecast_true, 
                           forecast_mask, 
                           max_forecast_horizon,
                           loss_fn,
                           dont_reduce=False):
    """
    Compute the variable forecasting loss.

    Note: batch_forecast_length can be less than or equal to max_forecast_horizon.
          If your true forecast is less than max_forecast_horizon, this function will pad.
          The prediction from the model will always be the max_forecast_horizon.

    Args:
        forecast_pred (torch.Tensor): Predicted forecasts of shape [batch_size, n_channels, max_forecast_horizon].
        forecast_true (torch.Tensor): True forecasts of shape [batch_size, n_channels, batch_forecast_length].
        forecast_mask (torch.Tensor): Mask indicating valid forecast values of shape [batch_size, n_channels, batch_forecast_length].
        loss_fn (callable): Loss function to use. Default is Mean Squared Error with no reduction.
        max_forecast_horizon (int): Maximum forecast horizon (specified by model settings).
        dont_reduce (bool): If True, returns the masked loss without reduction. Be aware that it will have zeros for masked out entries. Default is False.
    Returns:
        torch.Tensor: Computed mean loss.
    """
    if forecast_true.shape != forecast_mask.shape:
        raise ValueError(f"forecast_pred shape {forecast_pred.shape} must match forecast_mask shape {forecast_mask.shape}.")

    if forecast_mask.sum() == 0:
        raise ValueError("forecast_mask must have at least one non-zero value.")

    if forecast_true.shape[-1] < max_forecast_horizon:
        # Pad forecast_true and forecast_mask to max_forecast_horizon
        pad_len = max_forecast_horizon - forecast_true.shape[-1]
        forecast_true = torch.nn.functional.pad(forecast_true, (0, pad_len), "constant", 0)
        forecast_mask = torch.nn.functional.pad(forecast_mask, (0, pad_len), "constant", 0)

    loss = loss_fn(forecast_pred, forecast_true)

    if loss.shape != forecast_mask.shape:
        raise ValueError("The given loss function must not reduce the loss (i.e., use reduction='none')." \
        "The output of the loss function must must be of the same shape as forecast_pred.")

    masked_loss = loss * forecast_mask
    
    if dont_reduce:
        return masked_loss
    return masked_loss.sum() / forecast_mask.sum() 
