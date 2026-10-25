from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import requests
from aiohttp import payload
from sentence_transformers import SentenceTransformer
from collections import Counter
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_STORE_PATH = (
    PROJECT_ROOT / "models" / "product_catalog_vector_store.joblib"
)

DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"


@lru_cache(maxsize=1)
def load_rag_resources():
    """Load the product vectors and embedding model once."""

    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Vector store not found: {VECTOR_STORE_PATH}. "
            "Run Notebook 5 first."
        )

    vector_store = joblib.load(VECTOR_STORE_PATH)

    required_keys = {
        "documents",
        "metadata",
        "embeddings",
    }

    missing = required_keys - set(vector_store.keys())

    if missing:
        raise ValueError(
            f"Vector store is missing keys: {sorted(missing)}"
        )

    documents = vector_store["documents"]
    metadata = vector_store["metadata"]
    embeddings = np.asarray(
        vector_store["embeddings"],
        dtype=np.float32,
    )

    if not (
        len(documents) == len(metadata) == len(embeddings)
    ):
        raise ValueError(
            "Vector store documents, metadata and embeddings "
            "must have matching lengths."
        )

    if len(documents) == 0:
        raise ValueError("Vector store has no product documents.")

    model_name = vector_store.get(
        "embedding_model_name",
        DEFAULT_EMBEDDING_MODEL,
    )

    embedding_model = SentenceTransformer(model_name)

    return vector_store, embedding_model


def _safe_metadata_value(value):
    """Convert NumPy values and missing values to JSON-friendly values."""

    if value is None:
        return None

    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None

    if isinstance(value, np.generic):
        return value.item()

    return value


def retrieve_products(question: str, top_k: int = 4) -> list[dict]:
    """Find the most similar product documents for a question."""

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")

    vector_store, embedding_model = load_rag_resources()

    documents = vector_store["documents"]
    metadata = vector_store["metadata"]

    embeddings = np.asarray(
        vector_store["embeddings"],
        dtype=np.float32,
    )

    query_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError("Stored embeddings must be a 2D matrix.")

    if embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError(
            "Query embedding dimensions do not match "
            "stored product embeddings."
        )

    # Stored and query embeddings are normalized,
    # so dot product gives cosine similarity.
    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][
        :min(top_k, len(documents))
    ]

    results = []

    for index in top_indices:
        clean_metadata = {
            key: _safe_metadata_value(value)
            for key, value in metadata[index].items()
        }

        results.append({
            "similarity": float(scores[index]),
            "document": documents[index],
            "metadata": clean_metadata,
        })

    return results

def _answer_looks_repetitive(answer: str) -> bool:
    tokens = answer.split()

    if len(tokens) < 8:
        return False

    counts = Counter(tokens)
    most_common_count = counts.most_common(1)[0][1]

    return (
        most_common_count >= 4 and most_common_count / len(tokens) >= 0.2
    )

def _format_product_answer(sources: list[dict]) -> str:

    if not sources:
        return "သက်ဆိုင်ရာ product အချက်အလက် မတွေ့ရှိပါ။"

    products = []

    for item in sources:
        metadata = item.get("metadata", {})

        if metadata:
            products.append(metadata)

    if not products:
        return "Product အသေးစိတ်အချက်အလတ် မရရှိပါ"


    def fmt(value):
        if value is None:
            return "မဖော်ပြထား"

        try:
            number = float(value)

            if number.is_integer():
                return str(int(number))

            return f"{number:.2f}".strip("0").strip(".")
        except (ValueError, TypeError):
            return str(value)

    lines = [
        f"သက်ဆိုင်ရာ product {len(products)} ခုကို တွေ့ရှိပါတယ်", ""
    ]

    for index, product in enumerate(products, start=1):
        lines.extend([
            f"{index}. {product.get('name', 'Unknown Product')}",
            f" Product ID : {fmt(product.get('product_id'))}",
            f" Brand: {fmt(product.get('brand'))}",
            f" Category: {fmt(product.get('category'))}",
            f" Price: {fmt(product.get('price'))}",
            f" stock: {fmt(product.get('stock'))}",
            f" Rating: {fmt(product.get('rating'))}/5"
        ])

    valid_prices = []

    for product in products:
        try:
            price = float(product["price"])
            valid_prices.append((price, product))
        except (KeyError, TypeError, ValueError):
            continue

    if valid_prices:
        price, cheapest = min(valid_prices, key=lambda item: item[0])

        lines.extend([
            "နှိုင်းယှဉ်ချက်",
            (f"-စျေးအနိမ့်ဆုံး"
             f"{cheapest.get('name', 'Unknown Product')}"
             f"(ID: {fmt(cheapest.get('product_id'))})"
             f"Price: {fmt(price)}"
             )])

    return "\n".join(lines)

def generate_rag_answer(question: str, top_k: int = 4) -> dict:
    """Retrieve catalog documents and use Ollama to answer."""

    sources = retrieve_products(question, top_k=top_k)
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    ollama_url = os.getenv("OLLAMA_URL","https://localhost:11434/api/generate")

    if not sources:
        return {
            "question" : question,
            "answer" : "သက်ဆိုင်ရာ product အချက်အလတ် မတွေ့ရှိပါ",
            "model_name" : model_name,
            "sources" : []
        }

    context_parts = []

    for item in sources:
        if not isinstance(item, dict):
            continue

        document = item.get("document")

        if document:
            context_parts.append(str(document).strip())

        context = "\n\n".join(context_parts)

        if not context:
            answer = _format_product_answer(sources)

            return {
                "question" : question,
                "answer" : answer,
                "model_name" : model_name,
                "sources" : sources
            }

    prompt = f"""
You are TechCart AI, an IT hardware product assistant.

Answer the user's question in clear, natural Burmese.
Use only the product information probided below.

Rules:
1. Never invent product details, price, stock, or ratings.
2. Do not repeat words or sentence unnecessarily.
3. Keep the answer concise and easy to understand.
4. If several products are relevant, list them separately.
5. Include product name, category, price, stock, and rating only when those fields are available.
6. If the information is insufficient, say so clearly.

PRODUCT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER in Burmese:
"""
    payload = {
        "model" : model_name,
        "prompt" : prompt,
        "stream" : False,
        "options" : {
            "temperature" : 0.1,
            "num_predict" : 200
        }
    }
    ollama_response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout = (10, 600))

    ollama_response.raise_for_status()
    response_data = ollama_response.json()
    generate_text = response_data.get("response", "")

    if isinstance(generate_text, str):
        answer = generate_text.strip()
    else:
        answer = ""

    if (not answer or _answer_looks_repetitive(answer)):
        answer = _format_product_answer(sources)

    return {
        "question" : question,
        "answer" : answer,
        "model_name" : response_data.get("model") or model_name,
        "sources" : sources
    }