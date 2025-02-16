import pandas as pd
import numpy as np
import logging
from parkinson_updrs_regression.data_preparation import load_data, preprocess_data
from parkinson_updrs_regression.feature_engineering import compute_feature_correlations, select_features, normalize_features
from parkinson_updrs_regression.model_training import train_lls, gradient_descent, evaluate_model
from parkinson_updrs_regression.utils import setup_logger, log_message

def main():
    """Main pipeline for the Parkinson's regression analysis."""
    
    # Setup logger
    setup_logger()

    # Load dataset
    log_message("Loading dataset...")
    df = load_data("data/parkinsons_updrs.csv")

    # Preprocess data
    log_message("Preprocessing dataset...")
    df_clean = preprocess_data(df)

    # Compute and visualize feature correlations
    log_message("Analyzing feature correlations...")
    compute_feature_correlations(df_clean)

    # Feature selection (remove collinear features)
    log_message("Selecting optimal features...")
    df_selected = select_features(df_clean)

    # Normalize features
    log_message("Normalizing features...")
    df_normalized = normalize_features(df_selected)

    # Shuffle and split data
    log_message("Shuffling and splitting data...")
    np.random.seed(101)  # Set seed for reproducibility
    df_shuffled = df_normalized.sample(frac=1, random_state=101).reset_index(drop=True)
    
    Ntr = int(len(df_shuffled) * 0.5)  # 50% training, 50% test
    train_df, test_df = df_shuffled[:Ntr], df_shuffled[Ntr:]

    # Drop 'total_UPDRS' as it's the target variable
    if 'total_UPDRS' in train_df.columns:
        X_train, X_test = train_df.drop(columns=['total_UPDRS']), test_df.drop(columns=['total_UPDRS'])
    else:
        X_train, X_test = train_df, test_df

    y_train, y_test = train_df['total_UPDRS'], test_df['total_UPDRS']

    # Train model using Linear Least Squares (LLS)
    log_message("Training model using LLS...")
    w_lls = train_lls(X_train, y_train)

    # Train model using Gradient Descent
    log_message("Training model using Gradient Descent...")
    w_gd = gradient_descent(X_train, y_train)

    # Evaluate both models
    log_message("Evaluating models...")
    evaluate_model(X_test, y_test, w_lls, method="LLS")
    evaluate_model(X_test, y_test, w_gd, method="Gradient Descent")

if __name__ == "__main__":
    main()
