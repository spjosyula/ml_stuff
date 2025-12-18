import numpy as np
import matplotlib.pyplot as plt
from attention import SingleHeadAttention

# Create attention layer
d_model = 64
attn = SingleHeadAttention(d_model)

# Create input where words have clear patterns
seq_len = 10
x = np.random.randn(seq_len, d_model)

# Make words similar in a pattern (for demonstration)
# Word 0 and word 5 are similar
# Word 2 and word 7 are similar
x[5] = x[0] + np.random.randn(d_model) * 0.1  # word 5 similar to word 0
x[7] = x[2] + np.random.randn(d_model) * 0.1  # word 7 similar to word 2

# Forward pass
output = attn.forward(x)

# Visualize attention weights
plt.figure(figsize=(10, 8))
plt.imshow(attn.attention_weights, cmap='viridis', aspect='auto')
plt.colorbar(label='Attention Weight')
plt.xlabel('Key Position (words being attended TO)', fontsize=12)
plt.ylabel('Query Position (words attending FROM)', fontsize=12)
plt.title('Attention Weight Matrix\n(Darker = More Attention)', fontsize=14)

# Add grid for readability
for i in range(seq_len + 1):
    plt.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
    plt.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)

plt.tight_layout()
plt.savefig('attention_weights.png', dpi=150, bbox_inches='tight')
print("Attention visualization saved as 'attention_weights.png'")

print("\n=== Attention Weight Matrix ===")
print("Shape:", attn.attention_weights.shape)
print("\nFirst 5 rows (rounded to 2 decimals):")
print(np.round(attn.attention_weights[:5], 2))

print("\n=== Key Insight ===")
print("Look for patterns in the matrix:")
print("- Diagonal elements: How much each word attends to itself")
print("- Off-diagonal: How much words attend to other words")
print("- Similar words (0&5, 2&7) should show higher attention to each other")
print("\nNote: With random init weights, patterns are weak.")
print("After training, you'd see clear patterns (e.g., verbs attending to nouns)!")

plt.show()
