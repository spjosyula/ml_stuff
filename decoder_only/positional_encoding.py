import numpy as np

class PositionalEncoding:
    def __init__(self, d_model, max_seq_len=5000):
        self.d_model = d_model  # same dimension as embeddings (must match to add them together)
        self.max_seq_len = max_seq_len  # longest sentence we'll ever see
        self.encoding = self._create_encoding()  # precompute all position patterns (saves time)

    def _create_encoding(self):
        """Creates the fixed sine/cosine position patterns"""
        pe = np.zeros((self.max_seq_len, self.d_model))  # one row per position, one col per dimension

        position = np.arange(0, self.max_seq_len).reshape(-1, 1)  # [0, 1, 2, 3, ...] as column vector
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))  # creates decreasing frequencies
        # div_term spreads positional information across embedding dimensions at different scales, so the model can understand both short-range and long-range order.

        pe[:, 0::2] = np.sin(position * div_term)  # even columns (0, 2, 4...) get sine waves
        pe[:, 1::2] = np.cos(position * div_term)  # odd columns (1, 3, 5...) get cosine waves

        return pe  # shape: (max_seq_len, d_model)

    def forward(self, seq_len):
        """
        seq_len: length of the current sentence
        returns: position encodings for this sentence, shape (seq_len, d_model)
        """
        return self.encoding[:seq_len, :]  # just grab first seq_len rows (no learning, just lookup)

    # No backward pass needed. These are fixed patterns, not trainable weights
