import pandas as pd


def load_data(path):
    """Load data from CSV or Excel file."""
    if path.endswith('.xlsx'):
        return pd.read_excel(path)
    return pd.read_csv(path)


def clean_data(df):
    """Drop unnecessary columns, fix types, remove nulls.
    
    Drops location/ID columns that add noise with no predictive value.
    Fixes Total Charges which IBM stores as string with whitespace.
    Saves cleaned (pre-encoding) version for Power BI and EDA use.
    """
    # Validate expected target column exists
    if 'Churn Value' not in df.columns and 'Churn' not in df.columns:
        raise ValueError("Dataset missing target column: expected 'Churn Value' or 'Churn'")
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
