import numpy as np

def compute_feature_correlations(df, target='total_UPDRS'):
    """
    Computes the absolute correlations among non-target features,
    then drops features with any correlation > 0.9.
    The target column is reattached afterward.
    """
    if target in df.columns:
        target_series = df[target]
        df_features = df.drop(columns=[target])
    else:
        df_features = df.copy()
        target_series = None

    corr_matrix = df_features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
    df_features = df_features.drop(columns=to_drop, errors='ignore')
    
    if target_series is not None:
        df_features[target] = target_series
    return df_features

def drop_specific_features(df, features_to_drop):
    """
    Drops specified features (e.g., 'Jitter:DDP', 'Shimmer:DDA') only if they exist.
    """
    for feature in features_to_drop:
        if feature in df.columns:
            df = df.drop(columns=[feature])
    return df
