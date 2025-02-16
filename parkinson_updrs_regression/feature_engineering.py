from sklearn.preprocessing import StandardScaler

def select_features(df):
    """
    Selects relevant features from the dataset.
    
    Args:
        df (pd.DataFrame): Cleaned dataset.
    
    Returns:
        pd.DataFrame, pd.Series: Features (X) and target (y).
    """
    # Define feature columns
    feature_columns = ['age', 'sex', 'Jitter(%)', 'Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
    
    # Define target column
    target_column = 'total_UPDRS'
    
    X = df[feature_columns]
    y = df[target_column]
    
    return X, y

def normalize_features(X):
    """
    Normalizes numerical features using StandardScaler.
    
    Args:
        X (pd.DataFrame): Feature matrix.
    
    Returns:
        pd.DataFrame: Normalized feature matrix.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled
