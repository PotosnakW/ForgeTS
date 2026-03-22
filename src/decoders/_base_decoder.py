import logging
from common._modules import IdentityLayer


class BaseDecoder:
    def __init__(self):
        pass

    def get_decoder(self, config):
        decoder_key = getattr(config, "encoder", "none")
    
        if decoder_key is None or str(decoder_key).lower() == "none":
            print("No encoder selected — using identity pass-through.")
            return IdentityLayer()
    
        else:
            raise ValueError(f"encoder '{config.decoder}' not recognised.")
