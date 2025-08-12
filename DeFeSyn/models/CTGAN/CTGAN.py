import torch, io, gzip, base64, hashlib
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
        self.weights = {}


    def train(self):
        """
        Train the CTGAN model on the provided data.
        """
        self.model.fit(
            train_data=self.data,
            discrete_columns=self.discrete_columns
        )
        self.weights = self.get_weights()

    def sample(self, num_samples):
        """
        Generate synthetic samples from the trained CTGAN model.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            pd.DataFrame: Generated synthetic samples.
        """
        return self.model.sample(num_samples)

    def get_weights(self):
        """
        Get the weights of the trained CTGAN model.

        Returns:
            dict: Weights of the CTGAN model.
        """
        # TODO: Save only generator, or also discriminator? If both, are we allowed to make changes to the ctgan library (License)?
        return {
            'generator': self.model._generator.state_dict()
        }

    def load_weights(self, weights):
        """
        Load weights into the CTGAN model.

        Args:
            weights (dict): Weights to load into the model.
        """
        if 'generator' in weights:
            self.model._generator.load_state_dict(weights['generator'])
        else:
            return

    def encode(self):
        """Return a JSON-serializable package containing the weights."""
        try:
            state_dict = self.weights['generator']
        except KeyError:
            return None
        cooked = {}
        for k, v in state_dict.items():
            if torch.is_tensor(v):
                t = v.detach().cpu()
                if torch.is_floating_point(t):
                    t = t.to(torch.float64)
                cooked[k] = t
            else:
                cooked[k] = v

        buffer = io.BytesIO()
        torch.save(cooked, buffer)
        raw = buffer.getvalue()
        compressed = gzip.compress(raw)
        encoded = base64.b64encode(compressed).decode('utf-8')
        checksum = hashlib.sha256(compressed).hexdigest()
        return {
            'weights': encoded,
            'checksum': checksum
        }

    def decode(self, encoded_state_dict):
        """Decode the state_dict from a JSON-serializable package."""
        encoded = encoded_state_dict['weights']
        checksum = encoded_state_dict['checksum']
        compressed = base64.b64decode(encoded.encode('utf-8'))
        if hashlib.sha256(compressed).hexdigest() != checksum:
            raise ValueError("Checksum mismatch. The weights may be corrupted.")

        buffer = io.BytesIO(gzip.decompress(compressed))
        state_dict = torch.load(buffer, map_location='cpu')
        return {'generator': state_dict}
