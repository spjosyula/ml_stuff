import numpy as np
import matplotlib.pyplot as plt
from multi_head_attention import MultiHeadAttention

# Setup
np.random.seed(42)
d_model = 64
n_heads = 4
seq_len = 10

x = np.random.randn(seq_len, d_model)

# Create both types
mha_full = MultiHeadAttention(d_model, n_heads, causal=False)
mha_causal = MultiHeadAttention(d_model, n_heads, causal=True)

# Use same weights
mha_causal.W_q = mha_full.W_q.copy()
mha_causal.W_k = mha_full.W_k.copy()
mha_causal.W_v = mha_full.W_v.copy()
mha_causal.W_o = mha_full.W_o.copy()

# Forward pass
mha_full.forward(x)
mha_causal.forward(x)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Non-causal (full attention)
im1 = axes[0].imshow(mha_full.attention_weights[0], cmap='viridis', aspect='auto')
axes[0].set_title('Non-Causal Attention\n(Encoder style: see all tokens)', fontsize=12)
axes[0].set_xlabel('Key position (attending TO)')
axes[0].set_ylabel('Query position (attending FROM)')
plt.colorbar(im1, ax=axes[0])

# Causal (masked attention)
im2 = axes[1].imshow(mha_causal.attention_weights[0], cmap='viridis', aspect='auto')
axes[1].set_title('Causal Attention\n(Decoder/GPT style: only see past)', fontsize=12)
axes[1].set_xlabel('Key position (attending TO)')
axes[1].set_ylabel('Query position (attending FROM)')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('causal_mask_comparison.png', dpi=150, bbox_inches='tight')
print("Saved visualization")

plt.show()
