import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routers import rag as rag_router


client = TestClient(app)


# --------------------------------------------------
# 1. Health endpoints
# --------------------------------------------------

def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert data["docs"] == "/docs"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["database"] == "connected"


# --------------------------------------------------
# 2. Product list and catalog endpoints
# --------------------------------------------------

def test_list_products():
    response = client.get(
        "/products/",
        params={"limit": 5, "offset": 0},
    )

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) <= 5


def test_catalog_endpoints():
    endpoints = [
        "/catalog/summary",
        "/catalog/count",
        "/catalog/categories",
        "/catalog/brands",
        "/catalog/count?category=SSD",
    ]

    for endpoint in endpoints:
        response = client.get(endpoint)

        assert response.status_code == 200, (
            f"{endpoint} failed: {response.text}"
        )

        assert isinstance(response.json(), dict)


# --------------------------------------------------
# 3. Product CRUD lifecycle
# --------------------------------------------------

def test_product_crud_lifecycle():
    product_id = f"API-TEST-{uuid.uuid4().hex[:8]}"

    payload = {
        "product_id": product_id,
        "name": "Automated Test NVMe SSD",
        "category": "SSD",
        "brand": "TechCartTest",
        "price": 89.99,
        "stock": 12,
        "rating": 4.5,
    }

    response = client.post(
        "/products/",
        json=payload,
    )

    assert response.status_code == 201, response.text

    try:
        # READ
        response = client.get(
            f"/products/{product_id}"
        )

        assert response.status_code == 200

        product = response.json()

        assert product["product_id"] == product_id
        assert product["name"] == payload["name"]
        assert product["price"] == 89.99
        assert product["stock"] == 12

        # UPDATE
        response = client.patch(
            f"/products/{product_id}",
            json={
                "price": 79.99,
                "stock": 20,
            },
        )

        assert response.status_code == 200, response.text

        updated = response.json()

        assert updated["price"] == 79.99
        assert updated["stock"] == 20

        # Verify updated values via another GET
        response = client.get(
            f"/products/{product_id}"
        )

        assert response.status_code == 200
        assert response.json()["price"] == 79.99
        assert response.json()["stock"] == 20

        # LIST with filters
        response = client.get(
            "/products/",
            params={
                "category": "SSD",
                "brand": "TechCartTest",
                "limit": 100,
            },
        )

        assert response.status_code == 200

        matching_ids = {
            item["product_id"]
            for item in response.json()
        }

        assert product_id in matching_ids

    finally:
        # Always clean up the demo product.
        delete_response = client.delete(
            f"/products/{product_id}"
        )

        assert delete_response.status_code in (204, 404)

    # Verify the demo product no longer exists.
    response = client.get(
        f"/products/{product_id}"
    )

    assert response.status_code == 404


def test_get_missing_product_returns_404():
    response = client.get(
        "/products/DOES-NOT-EXIST-999999"
    )

    assert response.status_code == 404


# --------------------------------------------------
# 4. ML prediction endpoint
# --------------------------------------------------

def test_ml_prediction():
    response = client.post(
        "/ml/predict-category",
        json={
            "brand": "Samsung",
            "price": 176.65,
            "stock": 16,
            "rating": 4.7,
        },
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["predicted_category"]
    assert data["model_name"]

    probabilities = data["probabilities"]

    if probabilities is not None:
        assert len(probabilities) > 0

        assert all(
            0.0 <= value <= 1.0
            for value in probabilities.values()
        )


def test_ml_rejects_invalid_rating():
    response = client.post(
        "/ml/predict-category",
        json={
            "brand": "Samsung",
            "price": 100,
            "stock": 10,
            "rating": 8,
        },
    )

    assert response.status_code == 422


# --------------------------------------------------
# 5. NLP prediction endpoint
# --------------------------------------------------

def test_nlp_prediction():
    response = client.post(
        "/nlp/predict-category",
        json={
            "product_name": (
                "Samsung 990 PRO 1TB NVMe M.2 SSD"
            ),
        },
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["predicted_category"]
    assert data["cleaned_text"]
    assert data["model_name"]


def test_nlp_rejects_empty_product_name():
    response = client.post(
        "/nlp/predict-category",
        json={
            "product_name": "   ",
        },
    )

    assert response.status_code == 422


# --------------------------------------------------
# 6. RAG retrieval endpoint
# --------------------------------------------------

def test_rag_search():
    response = client.post(
        "/rag/search",
        json={
            "question": "Samsung SSD products",
            "top_k": 2,
        },
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["question"] == "Samsung SSD products"
    assert isinstance(data["results"], list)
    assert len(data["results"]) <= 2

    if data["results"]:
        result = data["results"][0]

        assert result["document"]
        assert isinstance(result["similarity"], (int, float))


def test_rag_search_rejects_invalid_top_k():
    response = client.post(
        "/rag/search",
        json={
            "question": "Samsung SSD products",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


# --------------------------------------------------
# 7. RAG generation API contract
# --------------------------------------------------

def test_rag_ask_endpoint(monkeypatch):

    def fake_generate_answer(question, top_k=4):
        return {
            "question": question,
            "answer": "Samsung SSD products were retrieved.",
            "model_name": "test-llm",
            "sources": [
                {
                    "similarity": 0.85,
                    "document": (
                        "Product ID: TEST-RAG-1\n"
                        "Product Name: Test SSD\n"
                        "Category: SSD\n"
                        "Brand: Samsung\n"
                        "Price: 100\n"
                        "Stock: 5\n"
                        "Rating: 4.5"
                    ),
                    "metadata": {
                        "product_id": "TEST-RAG-1",
                        "name": "Test SSD",
                        "category": "SSD",
                        "brand": "Samsung",
                        "price": 100.0,
                        "stock": 5,
                        "rating": 4.5,
                    },
                }
            ],
        }

    # Don't call the actual Ollama LLM in automated tests.
    monkeypatch.setattr(
        rag_router,
        "generate_rag_answer",
        fake_generate_answer,
    )

    response = client.post(
        "/rag/ask",
        json={
            "question": "Samsung SSD products",
            "top_k": 2,
        },
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["answer"]
    assert data["model_name"] == "test-llm"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["product_id"] == "TEST-RAG-1"