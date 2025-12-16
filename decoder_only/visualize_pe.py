
#Each horizontal row represents a position in a sentence (first word, second word, third word, etc.).
#Every row looks different on purpose and that difference is like a unique fingerprint for that position.
#The colors are just numbers going up and down smoothly. Nearby rows look similar, and rows that are far apart look very different.
#This helps the computer understand which words are close to each other and which are far apart.

import matplotlib.pyplot as plt
from positional_encoding import PositionalEncoding

# Create positional encoding
d_model = 128  # dimension (using 128 to see clear patterns)
max_len = 100  # look at first 100 positions
pe = PositionalEncoding(d_model, max_len)

# Get the encodings
encodings = pe.forward(max_len)  # shape: (100, 128)

print(f"Positional Encoding shape: {encodings.shape}")
print(f"Each row = position pattern for one word location in sentence\n")

# Visualize the pattern
plt.figure(figsize=(12, 8))
plt.imshow(encodings, cmap='RdBu', aspect='auto', interpolation='nearest')
plt.colorbar(label='Value')
plt.xlabel('Dimension (0 to d_model)', fontsize=12)
plt.ylabel('Position in sentence', fontsize=12)
plt.title('Positional Encoding Pattern\n(Each row = unique fingerprint for a position)', fontsize=14)
plt.tight_layout()
plt.savefig('positional_encoding_pattern.png', dpi=150, bbox_inches='tight')
print("Visualization saved as 'positional_encoding_pattern.png'")
plt.show()

# Show how different positions have unique patterns
print("\n=== Understanding the Pattern ===")
print(f"Position 0 encoding (first 10 dims): {encodings[0, :10]}")
print(f"Position 1 encoding (first 10 dims): {encodings[1, :10]}")
print(f"Position 50 encoding (first 10 dims): {encodings[50, :10]}")
print("\nNotice: Each position has a unique pattern!")
print("The model learns to recognize 'this word is at position X' from these patterns.")
