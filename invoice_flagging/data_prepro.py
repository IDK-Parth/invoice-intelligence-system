from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import sqlite3


def load_invoice_data(file_path):

    conn = sqlite3.connect(file_path)

    query = """
    WITH purchase_agg AS (
        SELECT
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )

    SELECT
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
        (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa
        ON vi.PONumber = pa.PONumber
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def create_invoice_risk_label(row):
    """
    Define invoice risk flagging logic based on row features.
    Returns 1 if invoice is flagged as risky, 0 otherwise.
    """
    # Example logic: flag if invoice dollars exceed item dollars or payment is delayed
    if pd.isna(row['invoice_dollars']) or pd.isna(row['total_item_dollars']):
        return 0
    
    if row['invoice_dollars'] > row['total_item_dollars'] * 1.1:
        return 1
    if row['days_to_pay'] > 30:
        return 1
    
    return 0


def apply_labels(df):
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df


def split_data(df, features, target):

    X = df[features]
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def scale_features(X_train, X_test, scaler_path):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled