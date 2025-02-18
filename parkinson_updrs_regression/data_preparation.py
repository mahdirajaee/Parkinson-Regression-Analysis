import pandas as pd
import numpy as np

def load_data(file_path="data/parkinsons_updrs.csv"):
    """Loads the dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Preprocesses the dataset:
      - Drops rows with missing values.
      - Removes 'subject#' column if it exists.
    """
    df = df.dropna()
    if 'subject#' in df.columns:
        df = df.drop(columns=['subject#'])
    return df

def shuffle_split_data(df, test_ratio=0.5, seed=123456):
    """
    Shuffles and splits the dataset into training and test sets.
    Default: 50% train, 50% test. Uses provided seed for reproducibility.
    """
    np.random.seed(seed)
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    N = len(df_shuffled)
    split_index = int(N * (1 - test_ratio))
    train_df = df_shuffled.iloc[:split_index].reset_index(drop=True)
    test_df  = df_shuffled.iloc[split_index:].reset_index(drop=True)
    return train_df, test_df

def normalize_features(df):
    """Normalizes all columns (zero mean, unit variance) using the dataframe's own stats."""
    return (df - df.mean()) / df.std()
