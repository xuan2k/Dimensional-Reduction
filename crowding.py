import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

# High-dimensional data
X = np.array([[0, 0], [0.5, 0.5], [1, 1], [15, 15], [15.5, 15.5]])
# X = np.array([[0, 0], [0.2, 0.2], [0.4, 0.4], [0.8, 0.8], [1.2, 1.2]])
# X = np.array([[0, 0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])

# SNE
sne = MDS(n_components=1, random_state=42)
Y_sne = sne.fit_transform(X)

# t-SNE
tsne = TSNE(n_components=1, random_state=42, perplexity=4)
Y_tsne = tsne.fit_transform(X)

# Plot the results
plt.figure(figsize=(15, 5))

# Original data points plot
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=['red', 'green', 'blue', 'orange', 'purple'], s=100)
plt.title('Original Data Points')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# SNE plot
plt.subplot(1, 3, 2)
plt.scatter(Y_sne, [0] * len(Y_sne), c=['red', 'green', 'blue', 'orange', 'purple'], s=100)
plt.title('SNE')
plt.xlabel('1D Embedding')
plt.yticks([])

# t-SNE plot
plt.subplot(1, 3, 3)
plt.scatter(Y_tsne, [0] * len(Y_tsne), c=['red', 'green', 'blue', 'orange', 'purple'], s=100)
plt.title('t-SNE')
plt.xlabel('1D Embedding')
plt.yticks([])

plt.tight_layout()
plt.show()
