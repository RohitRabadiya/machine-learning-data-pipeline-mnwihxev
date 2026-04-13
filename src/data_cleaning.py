import pandas as pd

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Fill missing values
    data = data.fillna(method='ffill')  # Forward fill
    
    return data