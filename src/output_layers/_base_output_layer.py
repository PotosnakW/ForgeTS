class BaseOutputLayer:
    def __init__(self):
        pass

    def get_output_layer(self, config):
        if config.output_layer.lower() == 'linear_proj':
            from output_layers.linear_proj import Linear_Multivariate_Layer
            output_layer = Linear_Multivariate_Layer(config)

        else:
            raise ValueError(
                f"output layer '{config.output_layer}' not recognised."
            )
            
        return output_layer