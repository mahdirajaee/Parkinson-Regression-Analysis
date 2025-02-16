import numpy as np

def train_lls(X_train, y_train):
    """Computes LLS regression coefficients."""
    return np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ y_train)

def train_steepest_descent(X_train, y_train, learning_rate=0.001, tol=1e-6, max_iter=1000):
    """Trains a regression model using steepest descent."""
    w = np.zeros(X_train.shape[1])
    for _ in range(max_iter):
        gradient = -2 * (X_train.T @ (y_train - X_train @ w)) / len(y_train)
        w -= learning_rate * gradient
        if np.linalg.norm(gradient) < tol:
            break
    return w

def predict(X, w):
    """Predicts the target variable using learned weights."""
    return X @ w
