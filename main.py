import numpy as np
import pandas as pd

from parkinson_updrs_regression.data_preparation import (
    load_data,
    preprocess_data,
    shuffle_split_data,
    normalize_features
)
from parkinson_updrs_regression.feature_engineering import (
    compute_feature_correlations,
    drop_specific_features
)
from parkinson_updrs_regression.model_training import (
    LLSRegression,
    SteepestDescentRegression
)
from parkinson_updrs_regression.utils import RegressionEvaluator

def main():
    # Use your ID/matricola as the seed
    ID_SEED = 123456

    # 1. Load dataset
    print("Loading dataset...")
    df = load_data("data/parkinsons_updrs.csv")

    # 2. Preprocess data
    print("Preprocessing dataset...")
    df_clean = preprocess_data(df)

    # 3. Feature engineering:
    #    - Exclude total_UPDRS from correlation-based dropping
    print("Analyzing feature correlations & removing highly correlated features (except target)...")
    df_corr = compute_feature_correlations(df_clean, target='total_UPDRS')

    # 4. Drop Jitter:DDP and Shimmer:DDA if they exist
    print("Dropping specific features: Jitter:DDP, Shimmer:DDA")
    df_final = drop_specific_features(df_corr, ["Jitter:DDP", "Shimmer:DDA"])

    # 5. Normalize features
    print("Normalizing features...")
    df_normalized = normalize_features(df_final)

    # 6. Shuffle & Split data
    print(f"Shuffling & splitting data (seed={ID_SEED})...")
    train_df, test_df = shuffle_split_data(df_normalized, test_ratio=0.5, seed=ID_SEED)

    # 7. Separate features & target
    #    Ensure total_UPDRS is still present after correlation-based removal
    if "total_UPDRS" not in train_df.columns:
        raise KeyError("total_UPDRS column is missing from the training set. Check your feature removal logic.")

    X_train = train_df.drop(columns=["total_UPDRS"]).values
    y_train = train_df["total_UPDRS"].values

    X_test = test_df.drop(columns=["total_UPDRS"]).values
    y_test = test_df["total_UPDRS"].values

    # 8. Train LLS
    print("\n--- Training LLS ---")
    lls = LLSRegression(X_train, y_train)
    lls.train()
    y_train_pred_lls = lls.predict(X_train)
    y_test_pred_lls  = lls.predict(X_test)

    # 9. Train Steepest Descent
    print("\n--- Training Steepest Descent ---")
    sgd = SteepestDescentRegression(X_train, y_train)
    sgd.train(learning_rate=0.001, tol=1e-6, max_iter=1000)
    y_train_pred_sgd = sgd.predict(X_train)
    y_test_pred_sgd  = sgd.predict(X_test)

    # 10. Evaluate & Compare
    print("\n--- LLS Results ---")
    RegressionEvaluator.print_results(y_train, y_train_pred_lls, y_test, y_test_pred_lls)

    print("\n--- Steepest Descent Results ---")
    RegressionEvaluator.print_results(y_train, y_train_pred_sgd, y_test, y_test_pred_sgd)

    # 11. Plot error histogram (for test set) and predictions
    error_lls_test = y_test - y_test_pred_lls
    RegressionEvaluator.plot_histogram(error_lls_test, title="LLS Test Error Histogram")

    RegressionEvaluator.plot_predictions(y_test, y_test_pred_lls, title="LLS Test: True vs Predicted")

if __name__ == "__main__":
    main()
