import logging
from common._modules import IdentityLayer

logger = logging.getLogger(__name__)

class BaseDecoder:
    def __init__(self):
        pass

    def get_decoder(self, config):
        decoder_key = getattr(config, "decoder", "none")   # was reading config.encoder
        if decoder_key is None or str(decoder_key).lower() == "none":
            logger.info("No decoder selected — using identity pass-through.")
            return IdentityLayer()
        else:
            raise ValueError(f"decoder '{config.decoder}' not recognised.")
