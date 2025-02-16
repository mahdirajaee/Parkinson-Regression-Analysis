import numpy as np
import pandas as pd

from parkinson_updrs_regression.data_preparation import load_data, preprocess_data, shuffle_split_data, normalize_features
from parkinson_updrs_regression.feature_engineering import compute_feature_correlations, drop_specific_features
from parkinson_updrs_regression.model_training import LLSRegression, SteepestDescentRegression, local_steepest_descent_regression
from parkinson_updrs_regression.utils import RegressionEvaluator

def main():
    # Use your matricola/ID as the seed (replace 123456 with your number)
    ID_SEED = 123456

    # 1. Load dataset
    print("Loading dataset...")
    df = load_data("data/parkinsons_updrs.csv")

    # 2. Preprocess dataset
    print("Preprocessing dataset...")
    df_clean = preprocess_data(df)

    # 3. Compute feature correlations and remove highly correlated features (excluding target)
    print("Computing feature correlations...")
    df_corr = compute_feature_correlations(df_clean, target='total_UPDRS')

    # 4. Drop specific features (if present)
    print("Dropping features: Jitter:DDP, Shimmer:DDA")
    df_final = drop_specific_features(df_corr, ["Jitter:DDP", "Shimmer:DDA"])

    # 5. Normalize the dataset
    print("Normalizing features...")
    df_normalized = normalize_features(df_final)

    # 6. Shuffle and split the dataset using your matricola as seed
    print(f"Shuffling and splitting data with seed {ID_SEED}...")
    train_df, test_df = shuffle_split_data(df_normalized, test_ratio=0.5, seed=ID_SEED)

    # 7. Separate features and target (ensure 'total_UPDRS' is present)
    if "total_UPDRS" not in train_df.columns:
        raise KeyError("total_UPDRS column is missing from the training set.")
    X_train = train_df.drop(columns=["total_UPDRS"]).values
    y_train = train_df["total_UPDRS"].values
    X_test = test_df.drop(columns=["total_UPDRS"]).values
    y_test = test_df["total_UPDRS"].values

    # 8. Global Regression Models

    print("\n--- Global LLS Regression ---")
    lls_model = LLSRegression(X_train, y_train)
    lls_model.train()
    y_train_pred_lls = lls_model.predict(X_train)
    y_test_pred_lls = lls_model.predict(X_test)

    print("\n--- Global Steepest Descent Regression ---")
    sgd_model = SteepestDescentRegression(X_train, y_train)
    sgd_model.train(learning_rate=0.001, tol=1e-6, max_iter=1000)
    y_train_pred_sgd = sgd_model.predict(X_train)
    y_test_pred_sgd = sgd_model.predict(X_test)

    print("\n--- Global LLS Results ---")
    RegressionEvaluator.print_results(y_train, y_train_pred_lls, y_test, y_test_pred_lls)

    print("\n--- Global Steepest Descent Results ---")
    RegressionEvaluator.print_results(y_train, y_train_pred_sgd, y_test, y_test_pred_sgd)

    # 9. Local Regression using Steepest Descent (using N nearest neighbors)
    print("\n--- Local Linear Regression (Steepest Descent, N=10) ---")
    y_test_pred_local = local_steepest_descent_regression(X_train, y_train, X_test,
                                                          N=10, learning_rate=0.001, tol=1e-6, max_iter=1000)
    mse_local, r2_local, corr_local = RegressionEvaluator.evaluate(y_test, y_test_pred_local)
    print("\nLocal Regression (N=10) Test Performance:")
    print(f"MSE: {mse_local:.4f}, R²: {r2_local:.4f}, Correlation: {corr_local:.4f}")

    RegressionEvaluator.plot_predictions(y_test, y_test_pred_local, title="Local Regression: True vs Predicted")
    RegressionEvaluator.plot_histogram(y_test - y_test_pred_local, title="Local Regression: Error Histogram")

if __name__ == "__main__":
    main()
