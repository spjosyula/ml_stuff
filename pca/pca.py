import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class SVD_PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components   #how many new simpler features we want to keep
        self.components_ = None    #the new directions/axes we'll use to look at the data
        self.explained_variance_ = None    #how much information each new direction captures
        self.explained_variance_ratio_ = None    #what percentage of total info each direction holds
        self.mean_ = None    #average value of each original feature (for centering)

    def fit(self, X):   #learn the new simpler directions from the data
        X = np.array(X)   
        n_samples, n_features = X.shape     #count how many data points and features we have
        
        self.mean_ = np.mean(X, axis=0)    #find average of each feature column
        X_centered = X - self.mean_    #subtract average so data is centered at zero
        
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)    #SVD: Math that finds best new directions
        
        explained_variance = (S ** 2) / (n_samples - 1)    #calculate how much info each direction holds
        
        if self.n_components is not None:   #if user wants only some directions
            Vt = Vt[:self.n_components, :]   #keep only the top requested directions
            explained_variance = explained_variance[:self.n_components]   #keep matching info amounts
        
        self.components_ = Vt   #save the new directions we found
        self.explained_variance_ = explained_variance   #save how much info each direction holds
        self.explained_variance_ratio_ = explained_variance / np.sum((S ** 2) / (n_samples - 1))   #convert to percentages
        
        return self   #return the fitted object so we can chain operations

    def transform(self, X):   #convert data to the new simpler coordinate system
        X = np.array(X)   #make sure data is in array format
        X_centered = X - self.mean_   #center the data like we did during training
        return X_centered @ self.components_.T   #rotate data into new directions

    def fit_transform(self, X):   #learn directions and convert data in one step
        self.fit(X)   #first learn the directions
        return self.transform(X)   #then convert the data

    def inverse_transform(self, X_transformed):   #convert simplified data back to original form
        return X_transformed @ self.components_ + self.mean_   #rotate back and add the average


def plot_variance(pca):   #show how much info each new direction captures
    n_components = len(pca.explained_variance_ratio_)   #count how many directions we have
    
    plt.figure(figsize=(12, 4))   #create a wide canvas for two plots
    
    plt.subplot(1, 2, 1)   #first plot on the left
    plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.7, color='steelblue')   #bar chart for each direction
    plt.xlabel('Principal Component')   #label x-axis
    plt.ylabel('Variance Ratio')   #label y-axis
    plt.title('Variance Explained by Each Component')   #title for the plot
    plt.xticks(range(1, n_components + 1))   #show numbers 1,2,3,4 on x-axis
    
    plt.subplot(1, 2, 2)   #second plot on the right
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)   #add up info as we go (running total)
    plt.plot(range(1, n_components + 1), cumulative_var, marker='o', color='darkblue')   #line plot with dots
    plt.xlabel('Number of Components')   #label x-axis
    plt.ylabel('Cumulative Variance Ratio')   #label y-axis (total info captured so far)
    plt.title('Cumulative Variance Explained')   #title for the plot
    plt.grid(True, alpha=0.3)   #add light grid lines
    plt.xticks(range(1, n_components + 1))   #show numbers 1,2,3,4 on x-axis
    
    plt.tight_layout()   #adjust spacing so labels don't overlap
    plt.show()   #display the plots

def plot_2d(X_transformed, y):   #visualize data in 2D using the new directions
    plt.figure(figsize=(8, 6))   #create a square-ish canvas
    scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1],    #plot first two new directions
                         c=y, cmap='viridis', alpha=0.7, edgecolors='black', s=50)   #color by class, add black outline
    plt.colorbar(scatter, label='Class')   #add color bar to show what colors mean
    plt.xlabel('PC1')   #first new direction (most important)
    plt.ylabel('PC2')   #second new direction (second most important)
    plt.title('2D PCA Projection')   #title for the plot
    plt.grid(True, alpha=0.3)   #add light grid lines
    plt.tight_layout()   #adjust spacing
    plt.show()   #display the plot

def demo():   #run a complete example showing how PCA works
    iris = load_iris()   #load the famous iris flower dataset
    X = iris.data   #get the measurements (features)
    y = iris.target   #get the flower types (labels)
    
    print("Dataset: Iris")   #print dataset name
    print(f"Shape: {X.shape}")   #show how many samples and features
    print(f"Features: {iris.feature_names}")   #show what we're measuring
    print(f"Classes: {iris.target_names}\n")   #show the flower types
    
    pca = SVD_PCA(n_components=None)   #create PCA object that keeps all directions
    pca.fit(X)   #learn the new directions from the data
    
    print("Explained Variance Ratio:")   #show how important each direction is
    for i, var_ratio in enumerate(pca.explained_variance_ratio_, 1):   #loop through each direction
        print(f"  PC{i}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")   #print percentage of info it holds
    
    print(f"\nCumulative Variance: {np.sum(pca.explained_variance_ratio_):.4f}\n")   #total should be 100%
    
    pca_2d = SVD_PCA(n_components=2)   #create new PCA that keeps only 2 directions
    X_2d = pca_2d.fit_transform(X)   #learn directions and convert data to 2D
    
    print(f"2D Shape: {X_2d.shape}")   #show new simplified shape
    print(f"Variance Captured: {np.sum(pca_2d.explained_variance_ratio_):.4f} ({np.sum(pca_2d.explained_variance_ratio_)*100:.2f}%)\n")   #show how much info we kept
    
    X_reconstructed = pca_2d.inverse_transform(X_2d)   #try to rebuild original data from 2D
    mse = np.mean((X - X_reconstructed) ** 2)   #calculate how different rebuilt data is
    print(f"Reconstruction Error (MSE): {mse:.6f}\n")   #print the error (smaller is better)
    
    plot_variance(pca)   #show bar charts of information captured
    plot_2d(X_2d, y)   #show 2D scatter plot with colors

if __name__ == "__main__":  
    demo()  
