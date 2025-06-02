import numpy as np

class NMF:
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
            # Store the old approximation for convergence check
            X_approx = W @ H

            # Update H
            numerator_H = W.T @ X
            denominator_H = (W.T @ W @ H) + eps
            H *= numerator_H / denominator_H

            # Update W
            numerator_W = X @ H.T
            denominator_W = (W @ H @ H.T) + eps
            W *= numerator_W / denominator_W

            # Check convergence (reconstruction error)
            reconstruction = W @ H
            error = np.linalg.norm(X - reconstruction, 'fro')

            if self.verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.max_iter}, error: {error:.4f}")

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

