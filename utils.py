import numpy as np
import matplotlib.pyplot as plt

def multiplicative_updates(X, n_components, max_iter=1500, W_init=None, H_init=None, random_state=0, tol=1e-4, verbose=False):
    rng = np.random.default_rng(random_state) # Initialize local random 
    m, n = X.shape
    r = n_components

    eps = 1e-10 # Small constant to avoid division by zero

    # Initialize W and H with values in [0, 1)
    W = W_init.copy() if W_init is not None else rng.random((m, r))
    H = H_init.copy() if H_init is not None else rng.random((r, n))

    # Rescale W to minimize ||X - WH||_F
    WH_init = W @ H
    numerator = np.sum(X * WH_init)
    denominator = np.maximum(np.sum(WH_init * WH_init), eps)
    alpha = numerator / denominator
    W *= alpha

    if verbose:
        print(f"Initial error: {np.linalg.norm(X - W @ H, 'fro'):.4f}")

    prev_error = None

    for i in range(max_iter):
        # Update rules for W and H
        W *= (X @ H.T) / np.maximum(W @ H @ H.T, eps)
        H *= (W.T @ X) / np.maximum(W.T @ W @ H, eps)

        X_reconstructed = W @ H
        error = np.linalg.norm(X - X_reconstructed, 'fro')
        
        if verbose and (i + 1) % 500 == 0:
            print(f"Iteration {i+1}/{max_iter}, Frobenius error: {error:.4f}")

        # --- Check for convergence ---
        if tol is not None and prev_error is not None:
            rel_change = abs(prev_error - error) / (prev_error + 1e-10)
            if rel_change < tol:
                if verbose:
                    print(f"Converged at iteration {i} with relative change {rel_change:.4e}")
                break

        prev_error = error

    return W, H



def hals_update(X, n_components, max_iter=1500, W_init=None, H_init=None, random_state=0, tol=1e-4, verbose=False):
    rng = np.random.default_rng(random_state)
    m, n = X.shape
    r = n_components

    eps = 1e-10 # Small constant to avoid division by zero

    # Initialize W and H with nonnegative random values
    W = W_init.copy() if W_init is not None else rng.random((m, r))
    H = H_init.copy() if H_init is not None else rng.random((r, n))

    # Rescale W to minimize ||X - WH||_F
    WH_init = W @ H
    numerator = np.sum(X * WH_init)
    denominator = np.maximum(np.sum(WH_init * WH_init), eps)
    alpha = numerator / denominator
    W *= alpha

    if verbose:
        print(f"Initial error: {np.linalg.norm(X - W @ H, 'fro'):.4f}")

    prev_error = None

    for i in range(max_iter):
        # --- Update H one row at a time ---
        WtW = W.T @ W
        WtX = W.T @ X
        for k in range(r):
            numerator = WtX[k, :] - WtW[k, :] @ H + WtW[k, k] * H[k, :]
            denominator = np.maximum(WtW[k, k], eps)
            H[k, :] = np.maximum(0, numerator / denominator)

        # --- Update W one column at a time ---
        HHt = H @ H.T
        XHt = X @ H.T
        for k in range(r):
            numerator = XHt[:, k] - W @ HHt[:, k] + HHt[k, k] * W[:, k]
            denominator = np.maximum(HHt[k, k], eps)
            W[:, k] = np.maximum(0, numerator / denominator)

        # --- Compute reconstruction error ---
        X_reconstructed = W @ H
        error = np.linalg.norm(X - X_reconstructed, 'fro')

        if verbose and (i + 1) % 500 == 0:
            print(f"Iteration {i+1}/{max_iter}, Frobenius error: {error:.4f}")

        # --- Check for convergence ---
        if tol is not None and prev_error is not None:
            rel_change = abs(prev_error - error) / (prev_error + 1e-10)
            if rel_change < tol:
                if verbose:
                    print(f"Converged at iteration {i} with relative change {rel_change:.4e}")
                break

        prev_error = error

    return W, H




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

def plot_matrix_histogram(A, name):
    a_min, a_max = A.min(), A.max()

    plt.figure(figsize=(8, 4))
    plt.hist(A.flatten(), bins=50, range=(a_min, a_max), color='goldenrod', edgecolor='black')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Values in {name}")
    plt.xlim(a_min, a_max)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()