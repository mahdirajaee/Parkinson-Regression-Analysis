import numpy as np

def compute_feature_correlations(df):
    """Computes correlation matrix and removes highly correlated features."""
    corr_matrix = df.corr().abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    return df.drop(columns=to_drop)
