from ..common._modules import IdentityLayer


class BaseInputLayer:
    def __init__(self):
        pass

    def get_input_layer(self, config):
        input_key = getattr(config, "input_layer", "none")

        if input_key is None or str(input_key).lower() == "none":
            print("No input layer selected — returning input as-is.")
            return IdentityLayer()
        
        elif config.input_layer.lower() == 'linear_proj':
            from .linear_proj import LinearProjectionLayer
            return LinearProjectionLayer(config)

        else:
            raise ValueError(
                f"input layer '{config.input_layer}' not recognised."
            )
