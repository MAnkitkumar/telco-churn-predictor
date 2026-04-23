import pandas as pd


def load_data(path):
    """Load data from CSV or Excel file."""
    if path.endswith('.xlsx'):
        return pd.read_excel(path)
    return pd.read_csv(path)


def clean_data(df):
    """Drop unnecessary columns, fix types, remove nulls."""
    # Drop columns not useful for modeling
    drop_cols = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
        'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 'Churn Score',
        'Churn Reason', 'CLTV'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Fix Total Charges — may have spaces or be non-numeric
    if 'Total Charges' in df.columns:
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
