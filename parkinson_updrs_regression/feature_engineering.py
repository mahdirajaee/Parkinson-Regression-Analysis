import numpy as np

def compute_feature_correlations(df):
    """Computes and visualizes feature correlations."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Matrix")
    plt.savefig("plots/feature_correlation.png")
    plt.show()

def select_features(df):
    """Removes highly correlated features."""
    correlation_matrix = df.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    high_correlation = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
    
    print(f"Removing highly correlated features: {high_correlation}")
    df = df.drop(columns=high_correlation, errors='ignore')
    return df

def normalize_features(df):
    """Normalizes features to have zero mean and unit variance."""
    df_normalized = (df - df.mean()) / df.std()
    return df_normalized
