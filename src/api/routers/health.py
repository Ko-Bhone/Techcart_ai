from fastapi import APIRouter
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from src.database.connection import engine
import logging

logger = logging.getLogger(__name__)
router = APIRouter(
    tags=["Health"]
)

@router.get("/")
def root():
    return {
        "message" : "Welcome to TechCart AI API...",
        "docs" : "/docs"
    }

@router.get("/health")
def health_check():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {
            "status" : "healthy",
            "database" : "connected"
        }

    except SQLAlchemyError as error:
        logger.exception("Health check failed")
        return {
            "status" : "unhealthy",
            "database" : "disconnected"
        }