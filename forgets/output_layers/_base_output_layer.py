from ..common._modules import IdentityLayer


class BaseOutputLayer:
    def __init__(self):
        pass

    def get_output_layer(self, config):
        output_key = getattr(config, "output_layer", "none")

        if output_key is None or str(output_key).lower() == "none":
            print("No output layer selected — returning encoder/decoder output as-is.")
            return IdentityLayer()
        
        elif config.output_layer.lower() == 'linear_proj':
            from .linear_proj import LinearProjectionLayer
            return LinearProjectionLayer(config)

        elif config.output_layer.lower() == 'linear_proj_multivariate':
            from .linear_proj import LinearProjectionLayerMultivariate
            return LinearProjectionLayerMultivariate(config)

        else:
            raise ValueError(
                f"output layer '{config.output_layer}' not recognised."
            )
