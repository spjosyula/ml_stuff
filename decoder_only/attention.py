import numpy as np

class SingleHeadAttention:
    def __init__(self, d_model):
        self.d_model = d_model  # dimension of input vectors
        self.d_k = d_model  # dimension for queries and keys (same as d_model for simplicity)

        # Weight matrices to project input into Query, Key, Value spaces
        self.W_q = np.random.randn(d_model, self.d_k) * 0.01  # transforms input to queries
        self.W_k = np.random.randn(d_model, self.d_k) * 0.01  # transforms input to keys
        self.W_v = np.random.randn(d_model, d_model) * 0.01  # transforms input to values

        # Store these for backward pass later
        self.Q = None  # queries: "what am I looking for?"
        self.K = None  # keys: "what do I contain?"
        self.V = None  # values: "what information do I have?"
        self.attention_weights = None  # softmax scores: "how much to focus on each word"
        self.x = None  # original input

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # subtract max for stability (prevents overflow)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)  # normalize to get probabilities that sum to 1

    def forward(self, x):
        """
        x: input tensor, shape (seq_len, d_model)
        returns: output after attention, shape (seq_len, d_model)
        """
        self.x = x  # save for backward pass

        # Step 1: Project input to Q, K, V
        self.Q = x @ self.W_q  # (seq_len, d_k) - what each word is searching for
        self.K = x @ self.W_k  # (seq_len, d_k) - what each word offers
        self.V = x @ self.W_v  # (seq_len, d_model) - actual information each word carries

        # Step 2: Calculate attention scores (how much each word should attend to others)
        scores = self.Q @ self.K.T  # (seq_len, seq_len) - dot product measures similarity
        scores = scores / np.sqrt(self.d_k)  # scale down to prevent extreme values (stabilizes gradients)

        # Step 3: Apply softmax to get attention weights (probabilities)
        self.attention_weights = self.softmax(scores)  # (seq_len, seq_len) - each row sums to 1

        # Step 4: Weighted sum of values
        output = self.attention_weights @ self.V  # (seq_len, d_model) - mix information based on attention

        return output  # each output word is now context-aware (knows about other words)
