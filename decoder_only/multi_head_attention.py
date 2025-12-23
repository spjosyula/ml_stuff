import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, n_heads, causal=False):
        self.d_model = d_model  # total embedding dimension
        self.n_heads = n_heads  # number of parallel attention heads
        self.head_dim = d_model // n_heads  # dimension per head
        self.causal = causal  # if True, mask future tokens (for decoder/GPT)

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Weight matrices (same total size as single-head, but will be split across heads)
        self.W_q = np.random.randn(d_model, d_model) * 0.01  # query projection
        self.W_k = np.random.randn(d_model, d_model) * 0.01  # key projection
        self.W_v = np.random.randn(d_model, d_model) * 0.01  # value projection
        self.W_o = np.random.randn(d_model, d_model) * 0.01  # output projection (combines all heads)

        # Store for backward pass
        self.x = None  # input
        self.Q = None  # queries before reshaping
        self.K = None  # keys before reshaping
        self.V = None  # values before reshaping
        self.Q_heads = None  # queries split into heads
        self.K_heads = None  # keys split into heads
        self.V_heads = None  # values split into heads
        self.attention_weights = None  # attention per head
        self.head_outputs = None  # output from each head before concat
        self.concat_output = None  # concatenated heads before final projection

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # numerical stability
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)  # probabilities sum to 1

    def softmax_backward(self, grad_output, softmax_output):
        """grad through softmax for each head"""
        sum_term = np.sum(softmax_output * grad_output, axis=-1, keepdims=True)
        return softmax_output * (grad_output - sum_term)

    def forward(self, x):
        """
        x: input, shape (seq_len, d_model)
        returns: output after multi-head attention, shape (seq_len, d_model)
        """
        self.x = x
        seq_len = x.shape[0]

        # Step 1: Project to Q, K, V
        self.Q = x @ self.W_q  # (seq_len, d_model)
        self.K = x @ self.W_k  # (seq_len, d_model)
        self.V = x @ self.W_v  # (seq_len, d_model)

        # Step 2: Split into multiple heads
        # Reshape: (seq_len, d_model) → (seq_len, n_heads, head_dim)
        self.Q_heads = self.Q.reshape(seq_len, self.n_heads, self.head_dim)
        self.K_heads = self.K.reshape(seq_len, self.n_heads, self.head_dim)
        self.V_heads = self.V.reshape(seq_len, self.n_heads, self.head_dim)

        # Transpose: (seq_len, n_heads, head_dim) → (n_heads, seq_len, head_dim)
        self.Q_heads = self.Q_heads.transpose(1, 0, 2)  # now each head processes independently
        self.K_heads = self.K_heads.transpose(1, 0, 2)
        self.V_heads = self.V_heads.transpose(1, 0, 2)

        # Step 3: Compute attention for each head in parallel
        scores = self.Q_heads @ self.K_heads.transpose(0, 2, 1)  # (n_heads, seq_len, seq_len)
        scores = scores / np.sqrt(self.head_dim)  # scale by sqrt of head dimension

        # Step 3.5: Apply causal mask (if enabled)
        if self.causal:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1)  # upper triangle (above diagonal) = 1
            scores = scores + (mask * -1e9)  # set future positions to -inf (use -1e9 for numerical stability)

        self.attention_weights = self.softmax(scores)  # (n_heads, seq_len, seq_len)

        # Step 4: Apply attention to values
        self.head_outputs = self.attention_weights @ self.V_heads  # (n_heads, seq_len, head_dim)

        # Step 5: Concatenate heads back together
        # Transpose: (n_heads, seq_len, head_dim) → (seq_len, n_heads, head_dim)
        head_outputs_transposed = self.head_outputs.transpose(1, 0, 2)

        # Reshape: (seq_len, n_heads, head_dim) → (seq_len, d_model)
        self.concat_output = head_outputs_transposed.reshape(seq_len, self.d_model)

        # Step 6: Final output projection
        output = self.concat_output @ self.W_o  # (seq_len, d_model)

        return output

    def backward(self, grad_output):
        """
        grad_output: gradient from next layer, shape (seq_len, d_model)
        returns: gradient wrt input x, shape (seq_len, d_model)
        """
        seq_len = self.x.shape[0]

        # Step 6 backward: output = concat_output @ W_o
        grad_concat = grad_output @ self.W_o.T  # (seq_len, d_model)
        self.grad_W_o = self.concat_output.T @ grad_output  # (d_model, d_model)

        # Step 5 backward: reshape and transpose back to heads
        grad_heads_transposed = grad_concat.reshape(seq_len, self.n_heads, self.head_dim)
        grad_head_outputs = grad_heads_transposed.transpose(1, 0, 2)  # (n_heads, seq_len, head_dim)

        # Step 4 backward: head_outputs = attention_weights @ V_heads
        grad_attention_weights = grad_head_outputs @ self.V_heads.transpose(0, 2, 1)  # (n_heads, seq_len, seq_len)
        grad_V_heads = self.attention_weights.transpose(0, 2, 1) @ grad_head_outputs  # (n_heads, seq_len, head_dim)

        # Step 3 backward: softmax
        grad_scores = self.softmax_backward(grad_attention_weights, self.attention_weights)  # (n_heads, seq_len, seq_len)

        # Backward through scaling
        grad_scores = grad_scores / np.sqrt(self.head_dim)

        # Backward through Q @ K.T
        grad_Q_heads = grad_scores @ self.K_heads  # (n_heads, seq_len, head_dim)
        grad_K_heads = grad_scores.transpose(0, 2, 1) @ self.Q_heads  # (n_heads, seq_len, head_dim)

        # Step 2 backward: merge heads back
        # Transpose: (n_heads, seq_len, head_dim) → (seq_len, n_heads, head_dim)
        grad_Q_heads = grad_Q_heads.transpose(1, 0, 2)
        grad_K_heads = grad_K_heads.transpose(1, 0, 2)
        grad_V_heads = grad_V_heads.transpose(1, 0, 2)

        # Reshape: (seq_len, n_heads, head_dim) → (seq_len, d_model)
        grad_Q = grad_Q_heads.reshape(seq_len, self.d_model)
        grad_K = grad_K_heads.reshape(seq_len, self.d_model)
        grad_V = grad_V_heads.reshape(seq_len, self.d_model)

        # Step 1 backward: Q, K, V projections
        self.grad_W_q = self.x.T @ grad_Q  # (d_model, d_model)
        self.grad_W_k = self.x.T @ grad_K  # (d_model, d_model)
        self.grad_W_v = self.x.T @ grad_V  # (d_model, d_model)

        # Gradient to input (sum gradients from all three projections)
        grad_x = grad_Q @ self.W_q.T + grad_K @ self.W_k.T + grad_V @ self.W_v.T

        return grad_x
