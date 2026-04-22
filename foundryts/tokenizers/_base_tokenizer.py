import logging
from ..common._modules import IdentityLayer

logger = logging.getLogger(__name__)

class BaseTokenizer:
    def __init__(self):
        pass

    def get_tokenizer(self, config):
        tokenizer_key = getattr(config, "tokenizer", "none")  
        if tokenizer_key is None or str(tokenizer_key).lower() == "none":
            logger.info("No tokenizer selected — using identity pass-through.")
            return IdentityLayer()
        
        elif tokenizer_key=='fixed_patch':
            from .patch import Patching
            return Patching(config)

        else:
            raise ValueError(f"tokenizer '{config.tokenizer}' not recognised.")
