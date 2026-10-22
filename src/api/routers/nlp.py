import re
from pathlib import Path
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from src.api.schemas import NLPTextPredictionRequest, NLPTextPredictionResponse


router = APIRouter(
    prefix = "/nlp",
    tags = ["NLP"]
    )

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "product_name_tfidf_classifier.joblib"

def clean_product_name(text: str) -> str:
    "Use the same text-cleaning steps as Notebook 4."

    text = text.lower()

    #Remove URLs
    text = re.sub(r"https?://\S+|ww\.\S+", " ", text)

    #Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    #keep letters, numbers, spaces, and selected symbols
    text = re.sub(r"[^\w\s\-\+\.]", " ", text)

    #Normalization repeated spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def load_nlp_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"NLP model not found: {MODEL_PATH}."
            "Run Notebook 4 and save the trained model first."
        )

    return joblib.load(MODEL_PATH)

try:
    nlp_model = load_nlp_model()
except Exception as err:
    raise RuntimeError(
        "Failed to load the NLP classifier"
        "Check the trained model file and Python environment"
    ) from err


@router.post("/predict-category", response_model = NLPTextPredictionResponse)

def predict_text_category(payload: NLPTextPredictionRequest):
    original_name = payload.product_name.strip()
    cleaned_text = clean_product_name(original_name)
    if not cleaned_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail = "Product name contains no usable text."
        )
    try:
        # The saved Pipeline performs TF-IDF transformation and classification together
        prediction = nlp_model.predict(cleaned_text)[0]
        probabilities = None
        if hasattr(nlp_model, "predict_proba"):
            raw_probabilities = nlp_model.predict_proba(cleaned_text)[0]

            classifier = nlp_model.named_steps["classifier"]
            classes = classifier.classes_
            probabilities = {
                str(label): float(probability)
                for label, probability in zip(classes, raw_probabilities)
            }

            probabilities = dict(sorted(probabilities.items(), key=lambda item:item[1], reverse=True))

        return NLPTextPredictionResponse(
            model_name = "TF-IDF + Logistic Regression",
            product_name = original_name,
            cleaned_text = cleaned_text,
            predicted_category = str(prediction),
            probabilities = probabilities,
        )

    except Exception as error:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = "NLP prediction failed. Check the API server logs"
        ) from error
