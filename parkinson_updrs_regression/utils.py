import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(y_true, y_pred):
    """Computes MSE, R², and correlation coefficient."""
    error = y_true - y_pred
    mse = np.mean(error**2)
    r2 = 1 - mse / np.var(y_true)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    return mse, r2, correlation

def plot_histogram(errors):
    """Plots histogram of regression errors."""
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=50, density=True, histtype='bar', label=['Training', 'Test'])
    plt.xlabel(r'$e=y-\hat{y}$')
    plt.ylabel(r'$P(e \in \text{bin})$')
    plt.legend()
    plt.grid()
    plt.title("Error Histogram")
    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred):
    """Plots true vs. predicted values."""
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, label="Predictions")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label="Ideal Fit")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid()
    plt.title("True vs. Predicted")
    plt.tight_layout()
    plt.show()

def print_results(y_train, y_train_pred, y_test, y_test_pred):
    """Prints performance metrics."""
    results = {
        "Training": evaluate_model(y_train, y_train_pred),
        "Test": evaluate_model(y_test, y_test_pred)
    }
    df_results = pd.DataFrame(results, index=['MSE', 'R²', 'Correlation'])
    print(df_results)
