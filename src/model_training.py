import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def train_model(df):
    """Train RandomForest with SMOTE balancing. Saves model + feature list."""
    target = 'Churn Value' if 'Churn Value' in df.columns else 'Churn'
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance — SMOTE on training set only
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE: {dict(zip(*__import__('numpy').unique(y_train, return_counts=True)))}")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'churn_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'features': list(X.columns)}, f)

    print(f"Model saved to {model_path}")
    return model, X_test, y_test
