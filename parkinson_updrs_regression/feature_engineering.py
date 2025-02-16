import numpy as np

def compute_feature_correlations(df, target='total_UPDRS'):
    """
    Computes the correlation among features (excluding the target) and drops
    those that are highly correlated (threshold > 0.9). The target column is kept.
    """
    if target in df.columns:
        target_series = df[target]
        df_features = df.drop(columns=[target])
    else:
        df_features = df.copy()
        target_series = None

    corr_matrix = df_features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    df_features = df_features.drop(columns=to_drop, errors='ignore')

    # Reattach the target column if it existed
    if target_series is not None:
        df_features[target] = target_series
    return df_features

def drop_specific_features(df, features_to_drop):
    """
    Drops specified features (e.g., 'Jitter:DDP' and 'Shimmer:DDA') only if they exist.
    """
    for feature in features_to_drop:
        if feature in df.columns:
            df = df.drop(columns=[feature])
    return df
