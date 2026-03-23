from types import SimpleNamespace

import torch
from torch import nn

from common._base_model import BaseModel
from common._modules import RevIN, Patching, PositionalEncoding, _make_causal_token_mask
from encoders._base_encoder import BaseEncoder
from output_layers._base_output_layer import BaseOutputLayer


class TransformerVLH(BaseModel):