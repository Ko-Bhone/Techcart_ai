from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.data.preprocessing import create_preprocessor
import joblib
import pandas as pd


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

def load_random_forest_model(model_path: str="models/random_forest_classifier.joblib"):
    return joblib.load(model_path)

def predict_product_category(model, product_data: pd.DataFrame):
    prediction = model.predict(product_data)
    return prediction

