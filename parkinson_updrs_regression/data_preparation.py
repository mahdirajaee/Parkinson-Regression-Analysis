import pandas as pd

def load_data(file_path="data/parkinsons_updrs.csv"):
    """Loads the dataset from CSV."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Ensure the dataset is in the correct directory.")
        exit(1)

def preprocess_data(df):
    """Preprocesses the dataset by handling missing values."""
    df = df.dropna()  # Remove rows with missing values
    return df
