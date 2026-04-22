import logging
from ..common._modules import IdentityLayer


class BaseEncoder:
    def __init__(self):
        pass

    def _get_huggingface_transformer(self, config):
        from transformers import T5Config
        from .t5_encoder import T5Model

        model_config = T5Config.from_pretrained(config.encoder)
        for attr in [
            "infini_mixer_type",
            "infini_channel_exclusion",
            "layerwise_beta",
            "channelwise_beta",
            "mlpmixer_hidden_size",
            "mlpmixer_n_layers",
            "mlpmixer_dropout",
        ]:
            setattr(model_config, attr, getattr(config, attr))

        transformer = T5Model(model_config)
        return transformer.get_encoder()

    def get_encoder(self, config):
        encoder_key = getattr(config, "encoder", "none")
    
        if encoder_key is None or str(encoder_key).lower() == "none":
            print("No encoder selected — using identity pass-through.")
            return IdentityLayer()
        
        elif config.encoder in [
            "google/t5-efficient-tiny",
            "google/t5-efficient-mini",
            "google/t5-efficient-small",
            "google/t5-efficient-base",
            "google/t5-efficient-large",
        ]:
            return self._get_huggingface_transformer(config)
    
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
