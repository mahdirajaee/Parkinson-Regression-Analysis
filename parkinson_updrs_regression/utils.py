import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class RegressionEvaluator:
    """Provides evaluation metrics and plotting utilities."""
    @staticmethod
    def evaluate(y_true, y_pred):
        error = y_true - y_pred
        mse = np.mean(error ** 2)
        var_y = np.var(y_true)
        r2 = 1 - mse / var_y if var_y != 0 else 0
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        return mse, r2, corr

    @staticmethod
    def plot_histogram(errors, title="Error Histogram"):
        plt.figure(figsize=(6, 4))
        plt.hist(errors, bins=50, density=True, histtype='bar')
        plt.xlabel(r'$e = y - \hat{y}$')
        plt.ylabel("Density")
        plt.title(title)
        plt.grid()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_predictions(y_true, y_pred, title="True vs Predicted"):
        plt.figure(figsize=(6, 4))
        plt.scatter(y_true, y_pred, alpha=0.7, label="Predictions")
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Fit")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_results(y_train, y_train_pred, y_test, y_test_pred):
        train_mse, train_r2, train_corr = RegressionEvaluator.evaluate(y_train, y_train_pred)
        test_mse, test_r2, test_corr = RegressionEvaluator.evaluate(y_test, y_test_pred)
        df_results = pd.DataFrame({
            "Training": [train_mse, train_r2, train_corr],
            "Test": [test_mse, test_r2, test_corr]
        }, index=["MSE", "R²", "Correlation"])
        print(df_results)
