from parkinson_updrs_regression.data_preparation import load_data, preprocess_data
from parkinson_updrs_regression.feature_engineering import select_features, normalize_features
from parkinson_updrs_regression.model_training import train_model, evaluate_model
from parkinson_updrs_regression.utils import setup_logger, log_message

def main():
    # Setup logger
    setup_logger()
    
    # Load dataset
    log_message("Loading dataset...")
    df = load_data("data/parkinsons_updrs.data")  # Update path if needed
    
    # Preprocess data
    log_message("Preprocessing dataset...")
    df_clean = preprocess_data(df)
    
    # Feature selection
    log_message("Selecting features...")
    X, y = select_features(df_clean)
    
    # Normalize features
    log_message("Normalizing features...")
    X_scaled = normalize_features(X)
    
    # Train model
    log_message("Training model...")
    model, X_test, y_test = train_model(X_scaled, y)
    
    # Evaluate model
    log_message("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Display results
    log_message(f"Model Performance: MAE={metrics['MAE']:.2f}, R²={metrics['R2 Score']:.2f}")

if __name__ == "__main__":
    main()
