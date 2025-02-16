import pandas as pd
import numpy as np

def load_data(file_path="data/parkinsons_updrs.csv"):
    """Loads and preprocesses the dataset."""
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """Cleans and prepares the dataset."""
    df.drop(columns=['subject#'], inplace=True)
    return df

def normalize_features(df):
    """Normalizes features to zero mean and unit variance."""
    return (df - df.mean()) / df.std()

def shuffle_split_data(df, test_ratio=0.5, seed=101):
    """Shuffles and splits data into training and test sets."""
    np.random.seed(seed)
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_index = int(len(df_shuffled) * (1 - test_ratio))
    return df_shuffled[:split_index], df_shuffled[split_index:]
