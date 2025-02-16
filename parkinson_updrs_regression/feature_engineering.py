import pandas as pd
import matplotlib.pyplot as plt

def compute_feature_correlations(df):
    """
    Computes correlation of all features with total UPDRS.

    Args:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        pd.Series: Correlation coefficients of features with total UPDRS.
    """
    corr_matrix = df.corr()
    total_updrs_corr = corr_matrix["total_UPDRS"].drop("total_UPDRS")  # Exclude target itself

    # Plot correlations
    plt.figure(figsize=(12, 6))
    total_updrs_corr.sort_values(ascending=False).plot(kind="bar")
    plt.title("Correlation Coefficients: Features vs Total UPDRS")
    plt.ylabel("Correlation")
    plt.xticks(rotation=90)
    plt.grid()
    plt.tight_layout()
    plt.show()

    return total_updrs_corr

def select_features(df, threshold=0.75):
    """
    Selects relevant features by removing highly correlated (collinear) ones.

    Args:
        df (pd.DataFrame): Cleaned dataset.
        threshold (float): Correlation threshold to remove redundant features.

    Returns:
        pd.DataFrame, pd.Series: Features (X) and target (y).
    """
    # Compute correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix to avoid duplicate pairs
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Removing highly correlated features: {to_drop}")

    # Define target variable
    target_column = "total_UPDRS"

    # Select final feature set
    X = df.drop(columns=to_drop + [target_column])
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
