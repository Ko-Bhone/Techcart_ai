from fastapi import APIRouter, HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from src.database.connection import engine

router = APIRouter(
    tags=["Health"]
)

@router.get("/")
def root():
    return {
        "Message" : "Welcome to TechCart AI API...",
        "docs" : "/docs"
    }

@router.get("/health")
def health_check():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {
            "status" : "healthy",
            "Database" : "Connected"
        }

    except SQLAlchemyError as error:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "Database Connection Failed"
        ) from error