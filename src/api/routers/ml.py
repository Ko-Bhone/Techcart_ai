import json
from pathlib import Path
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from src.api.schemas import MLPredictionRequest, MLPredictionResponse



router = APIRouter(
    prefix = "/ml",
    tags = ["Machine learning"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
METADATA_PATH = MODELS_DIR / "classification_metadata.json"
MODEL_FILES = {
    "Logistic Regression" : (MODELS_DIR / "logistic_regression_classifier.joblib"),
    "Random Forest" : (MODELS_DIR / "random_forest_classifier.joblib")
}


def load_classifier():
    selected_model_name = ("Random Forest")

    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r', encoding="utf-8") as file:
            metadata = json.load(file)
        selected_model_name = metadata.get("selected_model", "Random Forest")
    model_path = MODEL_FILES.get(selected_model_name)

    if model_path is None or not model_path.exists():
        available_models = [(name, path)
                            for name, path in MODEL_FILES.items()
                            if path.exists()]

        if not available_models:
            raise RuntimeError(f"No classification model found in {MODELS_DIR}")

        selected_model_name, model_path = available_models[0]
    model = joblib.load(model_path)
    return model, selected_model_name

try:
    classifier, classifier_name = load_classifier()
except Exception as error:
    raise RuntimeError(
        "Unable to load the trained ML classifier"
        "Check the models/ directory and classification_metadata.json"
    ) from error

@router.post("/predict-category", response_model=MLPredictionResponse)
def predict_category(payload: MLPredictionRequest):
    try:
        input_df = pd.DataFrame([{
            "brand" : payload.brand,
            "price" : payload.price,
            "stock" : payload.stock,
            "rating" : payload.rating
        }])
        prediction = classifier.predict(input_df)[0]
        probabilities = None

        if hasattr(classifier, "predict_proba"):
            raw_probabilities = classifier.predict_proba(input_df)[0]

            final_estimator = classifier.named_steps["classifier"]
            class_labels = final_estimator.classes_

            probabilities = {
                str(label) : float(probability)
                for label, probability in zip(class_labels, raw_probabilities)
            }

        return MLPredictionResponse(
            model_name = classifier_name,
            predicted_category = str(prediction),
            probabilities = probabilities
        )

    except Exception as error:
        raise HTTPException(
            status_code = 500,
            detail = "ML prediction failed, Check the server logs."
        ) from error