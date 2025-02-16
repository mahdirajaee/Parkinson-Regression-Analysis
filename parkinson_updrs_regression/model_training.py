import numpy as np

def train_lls(X_train, y_train):
    """Trains a model using Linear Least Squares (LLS)."""
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    
    w = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ y_train)
    return w

def gradient_descent(X_train, y_train, learning_rate=0.01, iterations=1000):
    """Trains a model using Gradient Descent."""
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    
    w = np.zeros(X_train.shape[1])
    for _ in range(iterations):
        gradient = (X_train.T @ ((X_train @ w) - y_train)) / len(y_train)
        w -= learning_rate * gradient
    return w

def evaluate_model(X_test, y_test, w, method="Model"):
    """Evaluates model performance using Mean Absolute Error and R² Score."""
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    
    y_pred = X_test @ w
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    print(f"{method} Performance: MAE={mae:.2f}, R²={r2:.2f}")
