from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.data.preprocessing import create_preprocessor
import joblib
import pandas as pd
from pathlib import Path

FEATURE_COLUMNS = ["brand", "price", "stock", "rating"]

def create_classification_pipeline():
    preprocessor = create_preprocessor()
    model = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline

def create_random_forest_pipeline():
    preprocessor = create_preprocessor()
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline

def load_random_forest_model(model_path = None):
    if model_path is None:
        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / "models" / "random_forest_classifier.joblib"
    return joblib.load(model_path)

def predict_product_category(model, product_data: pd.DataFrame):
    missing_columns = [column for column in FEATURE_COLUMNS
                       if column not in product_data.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    product_data = product_data[FEATURE_COLUMNS]
    prediction = model.predict(product_data)

    return prediction

