from ctgan import CTGAN

class CTGANModel:
    """
    CTGANModel is a wrapper for the CTGAN synthesizer from the ctgan library.
    """
    def __init__(self, data, discrete_columns, epochs, verbose=True):
        """
        Initialize the CTGAN model.

        Args:
            data (pd.DataFrame): The training data.
            discrete_columns (list): List of discrete columns in the data.
            epochs (int): Number of training epochs.
        """
        self.data = data
        self.discrete_columns = discrete_columns
        self.epochs = epochs
        self.verbose = verbose
        self.model = CTGAN(
            epochs=self.epochs,
            verbose=self.verbose
        )


    def train(self):
        """
        Train the CTGAN model on the provided data.
        """
        self.model.fit(
            train_data=self.data,
            discrete_columns=self.discrete_columns
        )

    def sample(self, num_samples):
        """
        Generate synthetic samples from the trained CTGAN model.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            pd.DataFrame: Generated synthetic samples.
        """
        return self.model.sample(num_samples)