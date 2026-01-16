import pandas as pd

TARGET_COL = "Transaction_Price"

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df
