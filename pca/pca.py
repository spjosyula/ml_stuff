import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class SVD_PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components   #how many principal components to keep
        self.components_ = None   #principal component directions (rows of Vt)
        self.explained_variance_ = None   #variance captured by each component
        self.explained_variance_ratio_ = None   #percentage of total variance each component explains
        self.mean_ = None   #mean of each feature (needed for centering)

    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        
        self.mean_ = np.mean(X, axis=0)   #compute mean of each feature column
        X_centered = X - self.mean_   #center data around zero (required for PCA)
        
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)   #SVD decomposes centered matrix into three parts
        
        explained_variance = (S ** 2) / (n_samples - 1)   #singular values squared give variance per component
        
        if self.n_components is not None:   #if limiting number of components
            Vt = Vt[:self.n_components, :]   #keep only top n_components rows
            explained_variance = explained_variance[:self.n_components]   #keep matching variances
        
        self.components_ = Vt   #each row is a principal component direction
        self.explained_variance_ = explained_variance   
        self.explained_variance_ratio_ = explained_variance / np.sum((S ** 2) / (n_samples - 1))   #convert to fraction of total
        
        return self

    def transform(self, X):
        X = np.array(X)
        X_centered = X - self.mean_   #center using the mean from training
        return X_centered @ self.components_.T   #project data onto principal components

    def fit_transform(self, X):
        self.fit(X)   #learn principal components
        return self.transform(X)   #then transform the data

    def inverse_transform(self, X_transformed):
        return X_transformed @ self.components_ + self.mean_   #project back to original space and add mean


iris = load_iris()
X = iris.data   #150 samples, 4 features (sepal/petal length/width)
y = iris.target   #3 classes (setosa, versicolor, virginica)

print(f"Shape: {X.shape}")
print(f"Features: {iris.feature_names}\n")

pca = SVD_PCA(n_components=None)   #keep all 4 components to see full breakdown
pca.fit(X)

print("Explained Variance Ratio:")
for i, var_ratio in enumerate(pca.explained_variance_ratio_, 1):
    print(f"  PC{i}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")

print(f"\nCumulative: {np.sum(pca.explained_variance_ratio_):.4f}\n")

pca_2d = SVD_PCA(n_components=2)   #reduce to 2 dimensions for visualization
X_2d = pca_2d.fit_transform(X)

print(f"2D Shape: {X_2d.shape}")
print(f"Variance Captured: {np.sum(pca_2d.explained_variance_ratio_):.4f}\n")

X_reconstructed = pca_2d.inverse_transform(X_2d)   #try to rebuild original 4D data from 2D
mse = np.mean((X - X_reconstructed) ** 2)   #measures information loss from dimensionality reduction
print(f"Reconstruction Error: {mse:.6f}\n")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.bar(range(1, 5), pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.title('Variance per Component')

plt.subplot(1, 2, 2)
cumulative_var = np.cumsum(pca.explained_variance_ratio_)   #cumulative sum shows how much total variance we capture
plt.plot(range(1, 5), cumulative_var, marker='o', color='darkblue')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
plt.title('Total Variance Captured')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolors='black', s=50)
plt.colorbar(scatter, label='Species')
plt.xlabel('PC1')   #PC1 captures most variance (72%)
plt.ylabel('PC2')   #PC2 captures second most variance (23%)
plt.title('2D PCA Projection')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()  
