import numpy as np

class RegressionModel:
    """Base class for regression models."""

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.w = None

    def predict(self, X):
        """Predicts target values using learned weights."""
        return X @ self.w

class LLSRegression(RegressionModel):
    """Linear Least Squares (LLS) Regression."""

    def train(self):
        """Computes LLS regression coefficients."""
        self.w = np.linalg.inv(self.X_train.T @ self.X_train) @ (self.X_train.T @ self.y_train)

class SteepestDescentRegression(RegressionModel):
    """Steepest Descent Regression with a stopping condition."""

    def train(self, learning_rate=0.001, tol=1e-6, max_iter=1000):
        """Trains using gradient descent with a stopping condition."""
        w = np.zeros(self.X_train.shape[1])
        for _ in range(max_iter):
            gradient = -2 * (self.X_train.T @ (self.y_train - self.X_train @ w)) / len(self.y_train)
            w -= learning_rate * gradient
            if np.linalg.norm(gradient) < tol:
                break
        self.w = w
