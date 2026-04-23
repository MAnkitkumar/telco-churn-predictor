import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import encode_data
from src.model_training import train_model
from src.evaluation import evaluate

# Run full pipeline: load → clean → encode → train → evaluate
# For multi-model comparison (LR vs RF vs XGBoost), run: python benchmark.py

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'Telco_customer_churn.xlsx')

df = load_data(DATA_PATH)
print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

df = clean_data(df)
print(f"After cleaning: {df.shape[0]} rows")

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'churn_cleaned.csv')
df.to_csv(PROCESSED_PATH, index=False)
print(f"Processed data saved to {PROCESSED_PATH}")

df = encode_data(df)
print("Encoding done.")

model, X_test, y_test = train_model(df)
evaluate(model, X_test, y_test)
