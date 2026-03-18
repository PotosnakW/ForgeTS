import logging
logger = logging.getLogger(__name__)

class BaseEncoder:
    def __init__(self):
        pass

    def _get_huggingface_transformer(self, config):
        from transformers import T5Config
        from encoders.t5_encoder import T5Model

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
        logger.info(f"Randomly initializing {config.encoder} ({T5Model.__name__}).")
        return transformer.get_encoder()

    def get_encoder(self, config):
        if config.encoder in [
            "google/t5-efficient-tiny",
            "google/t5-efficient-mini",
            "google/t5-efficient-small",
            "google/t5-efficient-base",
            "google/t5-efficient-large",
        ]:
            encoder = self._get_huggingface_transformer(config)
        elif config.encoder == "patchtst":
            from encoders.tst_encoder import TSTEncoder
            encoder = TSTEncoder(config)
        else:
            raise ValueError(f"encoder '{config.encoder}' not recognised.")

        return encoder