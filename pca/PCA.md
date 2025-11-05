# PCA and SVD: 

## What is PCA?

**Principal Component Analysis (PCA)** is a technique to simplify complex data by finding new, better ways to look at it.

### Idea

Imagine you have a dataset with many features (like height, weight, age, shoe size). Some of these features might be related like height and shoe size often go together. PCA finds new "directions" in your data that capture the most important patterns, letting you throw away the less important stuff.

## Example

Let's say you're measuring students:
- Feature 1: Height (in cm)
- Feature 2: Weight (in kg)
- Feature 3: Shoe size
- Feature 4: Age

You notice that height, weight, and shoe size are all related. Taller people tend to be heavier with bigger shoes. Instead of keeping all three, PCA might find one new feature called "Body Size" that captures most of this information.

## The PCA Algorithm

### Step 1: Center Your Data

Subtract the average from each feature so everything is centered at zero.

**Why?** We want to find patterns around the center, not be distracted by where the data sits.

```
Original height: [160, 170, 180]
Average: 170
Centered: [-10, 0, 10]
```

### Step 2: Find the Best Direction

Look for the direction where your data is most spread out (has the most variance).

**Why?** The direction with the most spread contains the most information. If all points line up perfectly along one line, you only need that line to describe them.

### Step 3: Find More Directions

Find the next best direction that is perpendicular to the first one. Keep going until you have as many directions as original features.

**Why?** Each new direction captures what's left over after the previous ones.

### Step 4: Keep Only the Important Ones

Usually, the first 2-3 directions capture most of the information (like 95%). Throw away the rest.

**Result:** You've gone from 4 features to 2 features, but kept 95% of the information.

## What is SVD?

**Singular Value Decomposition (SVD)** is a mathematical operation that breaks down a matrix into three simpler pieces. It's one way to actually compute PCA.

### The Math Behind It

Any matrix X can be broken down into:

```
X = U × S × V^T
```

Where:
- **U**: Left directions (in sample space) - not used much in PCA: Describes how each data point is positioned or expressed in terms of the main patterns found in the data.
- **S**: How important each direction is (diagonal matrix of singular values): Tells how powerful or significant each of those main patterns is. Bigger values mean stronger influence.
- **V**: Right directions (in feature space) - these are your principal components: Shows what those main patterns actually are, the directions or combinations of features that capture the most variation.

### Why Use SVD for PCA?

SVD directly gives us the principal components (the V matrix) without having to:
1. Calculate the covariance matrix
2. Find eigenvalues and eigenvectors

It's faster, more stable, and more accurate, especially for large datasets.

## The SVD-PCA Algorithm

### Step 1: Center the Data

```python
mean = average of each column
X_centered = X - mean
```

### Step 2: Apply SVD

```python
U, S, Vt = SVD(X_centered)
```

The Vt matrix rows are your principal components (new directions).

### Step 3: Calculate Explained Variance

```python
variance = (S^2) / (n_samples - 1)
```

Each singular value tells you how much information that direction captures.

### Step 4: Transform the Data

To convert your data to the new coordinate system:

```python
X_new = X_centered × V^T
```

This rotates your data into the new directions.

### Step 5: Reconstruct (Optional)

To go back to the original space:

```python
X_reconstructed = X_new × V + mean
```

If you used fewer components, this won't be perfect and you'll lose some information.

## Key Intuitions

### 1. Rotation

PCA is just rotating your data to a better angle. Like rotating a tilted ellipse so it aligns with the axes, makes it easier to describe.

### 2. Information vs Dimensions

More dimensions doesn't always mean more information. If two features always move together, they're really just one piece of information in disguise.

### 3. Variance = Information

The more spread out (variance) the data is in a direction, the more information that direction contains. Directions where data barely changes are considered unimportant.

### 4. Perpendicular is Key

Each principal component is perpendicular to all others. This means they capture completely different patterns - no overlap.

## When to Use PCA

**Good for:**
- Visualizing high-dimensional data (reduce to 2D or 3D)
- Removing noise from data
- Speeding up machine learning (fewer features = faster training)
- Finding patterns in data

**Not good for:**
- When you need to interpret what each feature means (new components are combinations of originals)
- When features have very different scales (always standardize first)
- When relationships are non-linear (PCA only finds straight-line patterns)

