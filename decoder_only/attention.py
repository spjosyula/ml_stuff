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

    def softmax_backward(self, grad_output):
        """
        grad_output: gradient flowing back from next layer, shape (seq_len, seq_len)
        returns: gradient wrt softmax input (scores), same shape

        Math: For softmax output S, gradient is S * (grad - sum(S * grad))
        Why? Because changing one input affects ALL outputs (they must sum to 1)
        """
        S = self.attention_weights  # softmax output we saved during forward, shape (seq_len, seq_len)

        # For each row: grad_input = S * (grad_output - sum(S * grad_output))
        sum_term = np.sum(S * grad_output, axis=-1, keepdims=True)  # weighted sum per row
        grad_scores = S * (grad_output - sum_term)  # element-wise: how much each score should change

        return grad_scores  # shape (seq_len, seq_len)

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

    def backward(self, grad_output):
        """
        grad_output: gradient from next layer, shape (seq_len, d_model)
        returns: gradient wrt input x, shape (seq_len, d_model)

        Chain rule: work backwards through each operation in forward pass
        """
        # Step 6 backward: output = attention_weights @ V
        grad_attention_weights = grad_output @ self.V.T  # how to change weights, shape (seq_len, seq_len)
        grad_V = self.attention_weights.T @ grad_output  # how to change V, shape (seq_len, d_model)

        # Step 5 backward: attention_weights = softmax(scores)
        grad_scores = self.softmax_backward(grad_attention_weights)  # shape (seq_len, seq_len)

        # Step 4 backward: scores = Q @ K.T / sqrt(d_k)
        grad_scores = grad_scores / np.sqrt(self.d_k)  # undo the scaling
        grad_Q = grad_scores @ self.K  # shape (seq_len, d_k)
        grad_K = grad_scores.T @ self.Q  # shape (seq_len, d_k)

        # Steps 1-3 backward: Q = x @ W_q, K = x @ W_k, V = x @ W_v
        self.grad_W_q = self.x.T @ grad_Q  # gradient for query weights
        self.grad_W_k = self.x.T @ grad_K  # gradient for key weights
        self.grad_W_v = self.x.T @ grad_V  # gradient for value weights

        # Gradient flows back through all three projections and sums up
        grad_x = grad_Q @ self.W_q.T + grad_K @ self.W_k.T + grad_V @ self.W_v.T

        return grad_x  # pass gradient to previous layer
