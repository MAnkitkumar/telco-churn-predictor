"""
Benchmark script: LR vs RF vs XGBoost with SMOTE + SHAP feature importance.
Prints all numbers needed for the README.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             classification_report)
from imblearn.over_sampling import SMOTE
import shap
import pickle

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import encode_data

# ── Load & prep ───────────────────────────────────────────────────────────────
DATA = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'Telco_customer_churn.xlsx')
df = encode_data(clean_data(load_data(DATA)))

target = 'Churn Value' if 'Churn Value' in df.columns else 'Churn'
X = df.drop(target, axis=1)
y = df[target]

print(f"\nClass distribution before SMOTE:\n{y.value_counts().to_dict()}")
print(f"Churn rate: {y.mean():.1%}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE on training set only
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"After SMOTE — train class distribution: {dict(zip(*np.unique(y_train_sm, return_counts=True)))}\n")

# ── Models ────────────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=100, random_state=42,
                                         eval_metric='logloss', verbosity=0),
}

results = {}
print(f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC-ROC':>9}")
print("-" * 75)

for name, clf in models.items():
    clf.fit(X_train_sm, y_train_sm)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    results[name] = {
        'accuracy':  accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall':    recall_score(y_test, preds),
        'f1':        f1_score(y_test, preds),
        'auc':       roc_auc_score(y_test, proba),
        'model':     clf,
    }
    r = results[name]
    print(f"{name:<25} {r['accuracy']:>9.1%} {r['precision']:>10.1%} "
          f"{r['recall']:>8.1%} {r['f1']:>8.1%} {r['auc']:>9.3f}")

# ── Best model = XGBoost (usually), save it ──────────────────────────────────
best_name = max(results, key=lambda k: results[k]['auc'])
best_model = results[best_name]['model']
print(f"\nBest model by AUC-ROC: {best_name}")

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, 'churn_model.pkl'), 'wb') as f:
    pickle.dump({'model': best_model, 'features': list(X.columns)}, f)
print(f"Best model saved.")

# ── SHAP feature importance (on RF for speed) ─────────────────────────────────
print("\n── SHAP Feature Importance (Random Forest) ──")
rf = results['Random Forest']['model']
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# For binary classification shap_values may be list [class0, class1]
sv = shap_values[1] if isinstance(shap_values, list) else shap_values
mean_shap = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
top5 = mean_shap.sort_values(ascending=False).head(5)
print(top5.to_string())

print("\n── Full Classification Report (Best Model) ──")
print(classification_report(y_test, best_model.predict(X_test)))
