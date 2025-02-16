import pandas as pd

def load_data(filepath):
    """
    Loads the dataset from a CSV file.
    
    Args:
        filepath (str): Path to the dataset file.
    
    Returns:
        pd.DataFrame: Loaded dataset as a Pandas DataFrame.
    """
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """
    Prepares the dataset by handling missing values and converting categorical variables.
    
    Args:
        df (pd.DataFrame): Raw dataset.
    
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        df = df.dropna()  # Drop missing values (optional: you can fill instead)
    
    # Convert categorical features (if applicable)
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(int)  # Convert 'sex' to integer
    
    return df
