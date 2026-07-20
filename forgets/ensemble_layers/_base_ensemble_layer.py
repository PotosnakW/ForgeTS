from ..common._modules import IdentityLayer

class BaseEnsembleLayer:
    def __init__(self):
        pass

    def get_ensemble_layer(self, config):
        input_key = getattr(config, "input_layer", "none")

        if input_key is None or str(input_key).lower() == "none":
            print("No input layer selected — returning input as-is.")
            return IdentityLayer()
        elif config.ensemble_layer.lower() != "none":
            from .ensembler_torch import Ensembler
            return Ensembler(config)
        else:
            raise ValueError(
                f"ensemble layer '{config.input_layer}' not recognised."
            )