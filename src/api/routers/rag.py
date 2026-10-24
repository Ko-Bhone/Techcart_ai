from fastapi import APIRouter, HTTPException, status
from requests.exceptions import RequestException
from src.api.schemas import RAGQueryRequest, RAGSource, RAGSearchResponse, RAGAnswerResponse
from src.rag.service import retrieve_products, generate_rag_answer
import logging

router = APIRouter(
    prefix="/rag",
    tags=["RAG"]
)

logger = logging.getLogger(__name__)

def format_source(result: dict) -> RAGSource:
    metadata = result["metadata"]

    return RAGSource(
        product_id = (
            str(metadata["product_id"])
            if metadata.get("product_id") is not None else None),

        name = metadata.get("name"),
        category = metadata.get("category"),
        brand = metadata.get("brand"),
        price = metadata.get("price"),
        stock = metadata.get("stock"),
        rating = metadata.get("rating"),
        similarity = round(float(result["similarity"]), 4),
        document = result["document"],
    )


@router.post("/search", response_model = RAGSearchResponse)
def search_catalog(payload: RAGQueryRequest):
    "Retrieve relevant catalog items without calling the LLM"

    try:
        results = retrieve_products(payload.question, top_k=payload.top_k)
        return RAGSearchResponse(
            question = payload.question,
            results = [format_source(item) for item in results])

    except FileNotFoundError as error:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = "RAG resources are invalid"
        ) from error

@router.post("/ask", response_model = RAGAnswerResponse)
def ask_catalog(payload: RAGQueryRequest):
    "Retrieve catalog items and generate answer using Ollama"
    try:
        result = generate_rag_answer(
            payload.question, top_k = payload.top_k
        )

        return RAGAnswerResponse(
            question = result["question"],
            answer = result ["answer"],
            model_name = result["model_name"],
            sources = [
                format_source(item) for item in result["sources"]
            ]
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "RAG vector store is unavailable"
        ) from error

    except RequestException as error:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = ("Ollama could not respond. Check that the server is running and the model is abailable")
        ) from error


    except ValueError as error:
        logger.exception("RAG / ask failed with ValueError")
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = "RAG resources are invalid"
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code = status.HTTP_502_BAD_GATEWAY,
            detail = "The RAG model did not generate a usable answer"
        ) from error

