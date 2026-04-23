from sklearn.preprocessing import LabelEncoder


def encode_data(df):
    """Label encode all categorical (object) columns.
    
    Uses alphabetical ordering (sklearn LabelEncoder default).
    The app/app.py mirrors this same ordering for inference.
    """
    le = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    return df
