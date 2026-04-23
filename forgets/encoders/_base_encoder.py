import logging
from ..common._modules import IdentityLayer


class BaseEncoder:
    def __init__(self):
        pass

    def get_encoder(self, config):
        encoder_key = getattr(config, "encoder", "none")
    
        if encoder_key is None or str(encoder_key).lower() == "none":
            print("No encoder selected — using identity pass-through.")
            return IdentityLayer()
    
        elif config.encoder == "patchtst":
            from .tst_encoder import TSTEncoder
            return TSTEncoder(config)
        
        elif config.encoder == "rnn":
            from .rnn_encoder import RNNEncoder
            return RNNEncoder(config)
    
        elif config.encoder == "lstm":
            from .lstm_encoder import LSTMEncoder
            return LSTMEncoder(config)
        
        elif config.encoder == "cnn":
            from .cnn_encoder import CNNEncoder
            return CNNEncoder(config)
    
        else:
            raise ValueError(f"encoder '{config.encoder}' not recognised.")
