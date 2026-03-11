import sqlite3
from sklearn.model_selection import train_test_split
import pandas as pd

def load_vendor_invoice_data(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM vendor_invoice", conn)
    conn.close()
    return df

def prepare_features(df: pd.DataFrame):
    X = df[["Dollars"]]
    y = df["Freight"]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)