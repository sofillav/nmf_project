import numpy as np
import matplotlib.pyplot as plt


class nmf:
    def __init__(self, n_components, max_iter=200, tol=1e-4, random_state=None, verbose=False):
        self.n_components = n_components  # Rank r
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def fit_transform(self, X):
        np.random.seed(self.random_state)
        m, n = X.shape
        r = self.n_components

        # Initialize W (m x r) and H (r x n) with small positive values
        W = np.random.rand(m, r)
        H = np.random.rand(r, n)

        eps = 1e-10  # Small constant to avoid division by zero

        for i in range(self.max_iter):
            # Update W
            numerator_W = X @ H.T
            denominator_W = np.maximum(W @ H @ H.T, eps)
            W *= numerator_W / denominator_W

            # Update H
            numerator_H = W.T @ X
            denominator_H = np.maximum(W.T @ W @ H, eps)
            H *= numerator_H / denominator_H

            # Compute reconstruction error
            reconstruction = W @ H
            error = np.linalg.norm(X - reconstruction, 'fro')

            # Print error
            if self.verbose and (i + 1) % 500 == 0:
                print(f"Iteration {i+1}/{self.max_iter}, error: {error:.4f}")

            # Check absolute tolerance
            if error < self.tol:
                break

        self.W = W
        self.H = H
        return W

    def inverse_transform(self):
        return self.W @ self.H

    def transform(self, X):
        # Not implemented: would require solving for H given W
        raise NotImplementedError("Use fit_transform to get W")



def plot_nmf_components(W, image_shape, n_components=None, n_cols=10, margin=2, title="NMF Basis Components"):
    """
    Plot the components W[:, j] as a tiled image grid (like a montage).
    
    Parameters:
        W             : np.ndarray, shape (h*w, r) - NMF basis matrix
        image_shape   : tuple, (h, w)
        n_components  : int or None - how many components to show (defaults to all)
        n_cols        : int - number of components per row
        margin        : int - pixels between images
        title         : str - title above the montage
    """
    h, w = image_shape
    r = W.shape[1]
    n_components = n_components or r
    n_rows = (n_components + n_cols - 1) // n_cols  # ceiling division

    # Create a blank canvas
    montage_height = n_rows * h + (n_rows - 1) * margin
    montage_width = n_cols * w + (n_cols - 1) * margin
    canvas = np.ones((montage_height, montage_width)) * 0.95  # light gray background

    for idx in range(n_components):
        row = idx // n_cols
        col = idx % n_cols
        comp = W[:, idx].reshape(h, w)
        comp = (comp - comp.min()) / (comp.max() - comp.min())  # normalize to [0, 1]

        y = row * (h + margin)
        x = col * (w + margin)
        canvas[y:y+h, x:x+w] = comp

    # Plot the montage
    plt.figure(figsize=(montage_width / 40, montage_height / 40))
    plt.imshow(canvas, cmap='gray', aspect='equal')
    plt.axis('off')
    plt.title(title, fontsize=12)
    plt.show()


def plot_reconstructed_images(X, X_reconstructed, image_shape, indices, n_cols=10, margin=2, title="Original vs Reconstructed Images"):
    """
    Plot original and reconstructed images as a montage grid.
    Each pair is shown vertically: original on top, reconstructed below.
    
    Parameters:
        X               : np.ndarray, shape (h*w, n_samples)
        X_reconstructed : np.ndarray, shape (h*w, n_samples)
        image_shape     : tuple, (h, w)
        indices         : list of sample indices to show
        n_cols          : int, number of image pairs per row
        margin          : int, pixels between images
        title           : str, title above the montage
    """
    h, w = image_shape
    n = len(indices)
    n_rows = (n + n_cols - 1) // n_cols  # number of rows of pairs

    # Total canvas size
    canvas_height = n_rows * 2 * h + (2 * n_rows - 1) * margin
    canvas_width = n_cols * w + (n_cols - 1) * margin
    canvas = np.ones((canvas_height, canvas_width)) * 0.95  # light gray background

    for idx, sample_idx in enumerate(indices):
        row = idx // n_cols
        col = idx % n_cols

        original = X[:, sample_idx].reshape(h, w)
        recon = X_reconstructed[:, sample_idx].reshape(h, w)

        # Normalize to [0, 1] using NumPy 2.0-safe approach
        original = (original - np.min(original)) / (np.ptp(original) + 1e-8)
        recon = (recon - np.min(recon)) / (np.ptp(recon) + 1e-8)

        y_top = row * (2 * h + margin)       # y position for original
        y_bottom = y_top + h + margin        # y position for reconstructed
        x = col * (w + margin)

        canvas[y_top:y_top+h, x:x+w] = original
        canvas[y_bottom:y_bottom+h, x:x+w] = recon

    # Show the montage
    plt.figure(figsize=(canvas_width / 40, canvas_height / 40))
    plt.imshow(canvas, cmap='gray', aspect='equal')
    plt.axis('off')
    plt.title(title, fontsize=12)
    plt.show()
