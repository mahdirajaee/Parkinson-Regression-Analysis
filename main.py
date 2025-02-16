import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from parkinson_updrs_regression.data_preparation import load_data, preprocess_data, shuffle_split_data
from parkinson_updrs_regression.feature_engineering import compute_feature_correlations, drop_specific_features
from parkinson_updrs_regression.model_training import LLSRegression, SteepestDescentRegression, local_steepest_descent_regression
from parkinson_updrs_regression.utils import RegressionEvaluator

def normalize_train_test(train_df, test_df, target='total_UPDRS'):
    """
    Normalizes train and test data using the training set's mean and std.
    Returns normalized train, normalized test, and (mu, sigma) for the target.
    """
    train_mean = train_df.mean()
    train_std = train_df.std()
    norm_train = (train_df - train_mean) / train_std
    norm_test = (test_df - train_mean) / train_std
    mu_target = train_mean[target]
    sigma_target = train_std[target]
    return norm_train, norm_test, mu_target, sigma_target

def denormalize(y_norm, mu, sigma):
    """Converts normalized values back to original scale."""
    return y_norm * sigma + mu

def compute_error_stats(errors):
    """Computes min, max, mean, std, and MSE of errors."""
    stats = {
        "min": np.min(errors),
        "max": np.max(errors),
        "mean": np.mean(errors),
        "std": np.std(errors),
        "MSE": np.mean(errors**2)
    }
    return stats

def main():
    # Use your matricola/ID as seed (replace 123456 with your actual ID)
    ID_SEED = 123456

    print("Loading dataset...")
    df = load_data("data/parkinsons_updrs.csv")

    print("Preprocessing dataset...")
    df_clean = preprocess_data(df)

    print("Computing feature correlations...")
    df_corr = compute_feature_correlations(df_clean, target='total_UPDRS')

    print("Dropping specific features: Jitter:DDP, Shimmer:DDA")
    df_final = drop_specific_features(df_corr, ["Jitter:DDP", "Shimmer:DDA"])

    # --- Split data before normalization ---
    print(f"Shuffling and splitting data with seed {ID_SEED}...")
    df_final = df_final.sample(frac=1, random_state=ID_SEED).reset_index(drop=True)
    N = len(df_final)
    split_index = int(N * 0.5)
    train_df = df_final.iloc[:split_index].reset_index(drop=True)
    test_df = df_final.iloc[split_index:].reset_index(drop=True)

    # --- Compute training target statistics ---
    mu_y = train_df["total_UPDRS"].mean()
    sigma_y = train_df["total_UPDRS"].std()

    # --- Normalize train and test using training stats ---
    norm_train, norm_test, mu_target, sigma_target = normalize_train_test(train_df, test_df, target="total_UPDRS")

    # Separate features and target from normalized data
    X_train = norm_train.drop(columns=["total_UPDRS"]).values
    y_train = norm_train["total_UPDRS"].values
    X_test = norm_test.drop(columns=["total_UPDRS"]).values
    y_test = norm_test["total_UPDRS"].values

    # --- Global Regression Models ---
    print("\n--- Global LLS Regression ---")
    lls_model = LLSRegression(X_train, y_train)
    lls_model.train()
    y_train_pred_lls = lls_model.predict(X_train)
    y_test_pred_lls  = lls_model.predict(X_test)

    print("\n--- Global Steepest Descent Regression ---")
    sgd_model = SteepestDescentRegression(X_train, y_train)
    sgd_model.train(learning_rate=0.001, tol=1e-6, max_iter=1000)
    y_train_pred_sgd = sgd_model.predict(X_train)
    y_test_pred_sgd  = sgd_model.predict(X_test)

    # --- Local Regression (using N=10 nearest neighbors) ---
    print("\n--- Local Regression (Steepest Descent, N=10) ---")
    y_test_pred_local = local_steepest_descent_regression(X_train, y_train, X_test,
                                                          N=10, learning_rate=0.001, tol=1e-6, max_iter=1000)

    # --- De-normalize predictions and true targets ---
    y_train_denorm = denormalize(y_train, mu_target, sigma_target)
    y_test_denorm  = denormalize(y_test, mu_target, sigma_target)
    y_train_pred_lls_denorm = denormalize(y_train_pred_lls, mu_target, sigma_target)
    y_test_pred_lls_denorm  = denormalize(y_test_pred_lls, mu_target, sigma_target)
    y_train_pred_sgd_denorm = denormalize(y_train_pred_sgd, mu_target, sigma_target)
    y_test_pred_sgd_denorm  = denormalize(y_test_pred_sgd, mu_target, sigma_target)
    y_test_pred_local_denorm = denormalize(y_test_pred_local, mu_target, sigma_target)

    # --- Plot: Estimated vs. True (Test) for Global LLS and Local Regression ---
    RegressionEvaluator.plot_predictions(y_test_denorm, y_test_pred_lls_denorm, 
                                           title="Global LLS: True vs Predicted (Test, De-normalized)")
    RegressionEvaluator.plot_predictions(y_test_denorm, y_test_pred_local_denorm, 
                                           title="Local Regression: True vs Predicted (Test, De-normalized)")

    # --- Plot: Histograms of estimation error (de-normalized) ---
    error_train_lls = y_train_denorm - y_train_pred_lls_denorm
    error_test_lls  = y_test_denorm - y_test_pred_lls_denorm
    error_test_local = y_test_denorm - y_test_pred_local_denorm

    RegressionEvaluator.plot_histogram(error_train_lls, title="Global LLS Training Error Histogram (De-normalized)")
    RegressionEvaluator.plot_histogram(error_test_lls, title="Global LLS Test Error Histogram (De-normalized)")
    RegressionEvaluator.plot_histogram(error_test_local, title="Local Regression Test Error Histogram (De-normalized)")

    # --- Build a metrics table for test dataset ---
    def build_metrics(y_true, y_pred):
        error = y_true - y_pred
        stats = {
            "min": np.min(error),
            "max": np.max(error),
            "mean": np.mean(error),
            "std": np.std(error),
            "MSE": np.mean(error**2)
        }
        mse, r2, corr = RegressionEvaluator.evaluate(y_true, y_pred)
        stats["R²"] = r2
        stats["Correlation"] = corr
        return stats

    metrics_lls = build_metrics(y_test_denorm, y_test_pred_lls_denorm)
    metrics_local = build_metrics(y_test_denorm, y_test_pred_local_denorm)

    df_metrics = pd.DataFrame({
        "Global LLS": metrics_lls,
        "Local Regression (N=10)": metrics_local
    })
    print("\n--- Performance Metrics (Test Dataset, De-normalized) ---")
    print(df_metrics)
    df_metrics.to_csv("performance_metrics.csv", index=True)

    # --- (Optional) Loop over 20 different seeds and average results (not implemented here) ---
    # You can run this script multiple times with different seeds, accumulate the metrics, and average them.

if __name__ == "__main__":
    main()
