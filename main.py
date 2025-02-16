import numpy as np
import pandas as pd
from parkinson_updrs_regression.data_preparation import (
    load_data, clean_data, normalize_features, shuffle_split_data
)
from parkinson_updrs_regression.feature_engineering import compute_feature_correlations
from parkinson_updrs_regression.model_training import train_lls, train_steepest_descent, predict
from parkinson_updrs_regression.utils import plot_histogram, plot_predictions, print_results

def main():
    # Load & Preprocess Data
    df = load_data()
    df_clean = clean_data(df)
    
    # Feature Engineering
    df_selected = compute_feature_correlations(df_clean)
    df_normalized = normalize_features(df_selected)

    # Split Data
    train_df, test_df = shuffle_split_data(df_normalized)

    # Separate Features & Target
    X_train, y_train = train_df.drop(columns=['total_UPDRS']).values, train_df['total_UPDRS'].values
    X_test, y_test = test_df.drop(columns=['total_UPDRS']).values, test_df['total_UPDRS'].values

    # Train LLS
    w_lls = train_lls(X_train, y_train)
    y_train_pred_lls = predict(X_train, w_lls)
    y_test_pred_lls = predict(X_test, w_lls)

    # Train Steepest Descent
    w_sgd = train_steepest_descent(X_train, y_train)
    y_train_pred_sgd = predict(X_train, w_sgd)
    y_test_pred_sgd = predict(X_test, w_sgd)

    # Evaluate & Plot Results
    print("\n--- LLS Regression Results ---")
    print_results(y_train, y_train_pred_lls, y_test, y_test_pred_lls)

    print("\n--- Steepest Descent Regression Results ---")
    print_results(y_train, y_train_pred_sgd, y_test, y_test_pred_sgd)

    plot_histogram(y_test - y_test_pred_lls)
    plot_predictions(y_test, y_test_pred_lls)

if __name__ == "__main__":
    main()
