import numpy as np

class RegressionModel:
    """Base class for regression models."""
    def __init__(self, X_train, y_train):
        self.X_train = X_train  # Should be a NumPy array of shape (n_train, d)
        self.y_train = y_train  # Should be a NumPy array of shape (n_train,)
        self.w = None

    def predict(self, X):
        """Predicts target values using learned weights."""
        return X @ self.w

class LLSRegression(RegressionModel):
    """Global Linear Least Squares Regression."""
    def train(self):
        self.w = np.linalg.inv(self.X_train.T @ self.X_train) @ (self.X_train.T @ self.y_train)

class SteepestDescentRegression(RegressionModel):
    """Global Regression using steepest descent."""
    def train(self, learning_rate=0.001, tol=1e-6, max_iter=1000):
        w = np.zeros(self.X_train.shape[1])
        for _ in range(max_iter):
            gradient = -2 * (self.X_train.T @ (self.y_train - self.X_train @ w)) / len(self.y_train)
            w_new = w - learning_rate * gradient
            if np.linalg.norm(w_new - w) < tol:
                w = w_new
                break
            w = w_new
        self.w = w

def local_steepest_descent_regression(X_train, y_train, X_test, N=10, learning_rate=0.001, tol=1e-6, max_iter=1000):
    """
    For each test point in X_test, performs a local linear regression using the N closest
    training points. Steepest descent is used to find the local weights.
    
    Parameters:
      X_train: NumPy array of shape (n_train, d)
      y_train: NumPy array of shape (n_train,)
      X_test: NumPy array of shape (n_test, d)
      N: Number of nearest neighbors to use (default: 10)
      learning_rate, tol, max_iter: Parameters for steepest descent.
      
    Returns:
      y_pred: NumPy array of predictions for the test set.
    """
    n_test = X_test.shape[0]
    y_pred = np.zeros(n_test)
    
    for i in range(n_test):
        x = X_test[i, :]  # Test point (1 x d)
        # Compute Euclidean distances from x to each training point
        distances = np.linalg.norm(X_train - x, axis=1)
        neighbor_idx = np.argsort(distances)[:N]  # Indices of the N closest points
        
        X_local = X_train[neighbor_idx, :]
        y_local = y_train[neighbor_idx]
        
        # Initialize local weights for steepest descent
        w_local = np.zeros(X_local.shape[1])
        for _ in range(max_iter):
            gradient = -2 * (X_local.T @ (y_local - X_local @ w_local)) / len(y_local)
            w_new = w_local - learning_rate * gradient
            if np.linalg.norm(w_new - w_local) < tol:
                w_local = w_new
                break
            w_local = w_new
        y_pred[i] = x @ w_local
    return y_pred
