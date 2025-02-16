from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(X, y):
    """
    Trains a Linear Regression model.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
    
    Returns:
        model: Trained regression model.
        X_test: Test feature data.
        y_test: Test target data.
    """
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model using MAE and R².
    
    Args:
        model: Trained regression model.
        X_test (pd.DataFrame): Test feature data.
        y_test (pd.Series): Test target data.
    
    Returns:
        dict: Evaluation metrics (MAE, R²)
    """
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {"MAE": mae, "R2 Score": r2}
