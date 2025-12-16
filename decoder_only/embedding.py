import numpy as np

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size  # total number of unique tokens (words) in your vocabulary
        self.d_model = d_model  # dimension of each embedding vector (how many numbers represent each word)
        self.weights = np.random.randn(vocab_size, d_model) * 0.01  # each row = embedding for one token, small random init
        self.indices = None  # stores which tokens/words were looked up (needed for backward pass)

    def forward(self, indices):
        """
        indices: array of token IDs, shape (batch_size, seq_len) or (seq_len,)
        returns: embeddings, shape (*indices.shape, d_model)
        """
        self.indices = indices  # token ids are saved for training
        return self.weights[indices]  # lookup: grab the rows corresponding to each token ID

    def backward(self, grad_output):
        """
        grad_output: gradient from next layer, same shape as forward output (how much each embedding contributes to the error)
        returns: None (we only update weights, no gradient flows back through indices)
        """
        grad_weights = np.zeros_like(self.weights)  # start with zero gradient for all embeddings
        np.add.at(grad_weights, self.indices, grad_output)  # accumulate gradients into rows that were actually used. If same word appears 3 times, its gradients are summed, not overwritten
        self.grad_weights = grad_weights  # store for optimizer to update weights
        # Note: No gradient wrt indices (they're integers, not differentiable)
