class Consensus:
    """
    A class to represent a consensus mechanism.
    """

    def __init__(self, model_type="CTGAN"):
        """
        Initialize the Consensus class.
        Args:
            model_type (str): Type of the model for which consensus is being calculated.
        """
        # TODO: Dynamic calculation of learning factor
        self.learning_factor = 0.5
        self.model_type = model_type

    def average(self, x_i: dict, x_j: dict):
        """
        Performs consensus averaging of two sets of model weights. The formula for the model update is:
        x_i = x_i + learning_factor * (x_j - x_i)
        Args:
            x_i (dict): Weights of the first model.
            x_j (dict): Weights of the second model.
        Returns:
            dict: Averaged weights.
        """
        # weight update: x_i = x_i + learning_factor * (x_j - x_i)
        if self.model_type == "CTGAN":
            # Assuming x_i and x_j are dictionaries with 'generator' key containing the weights
            gen_i = x_i['generator']
            gen_j = x_j['generator']
            dis_i = x_i['discriminator']
            dis_j = x_j['discriminator']
            averaged_weights = {
                'generator': {key: value + self.learning_factor * (gen_j[key] - value) for key, value in gen_i.items()},
                'discriminator': {key: value + self.learning_factor * (dis_j[key] - value) for key, value in dis_i.items()}
            }
            return averaged_weights
        else:
            raise ValueError(f"Model type {self.model_type} not supported for consensus averaging.")