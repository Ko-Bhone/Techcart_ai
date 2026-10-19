from collections.abc import Generator
from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal, engine
from src.database.crud import create_product, get_product, list_products, update_product, delete_product
from src.api.schemas import ProductCreate, ProductUpdate, ProductRead

app = FastAPI(
    title = "TechCart AI API",
    description = "Product Management API for TechCart AI",
    version = "1.0.0"
)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {
        "Message": "Welcome to TechCart AI API",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {
            "status" : "healthy",
            "database" : "connected"
        }

    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database Connection Failed"
        ) from error


@app.get("/products", response_model=list[ProductRead])
def get_products(
        category: str | None = None,
        brand : str | None = None,
        limit : int = Query(default=100, ge=1, le=500),
        offset : int = Query(default=0, ge=0),
        db: Session = Depends(get_db)
        ):
    return list_products(
        db,
        category = category,
        brand = brand,
        limit = limit,
        offset = offset
    )

@app.get("/products/{product_id", response_model=ProductRead)
def get_product_by_id(product_id : str, db: Session = Depends(get_db)):
    product = get_product(db, product_id)

    if product is None:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "Product Not Found"
        )
    return product

@app.post("/products", response_model=ProductRead, status_code=status.HTTP_201_CREATED)
def add_product(payload: ProductCreate, db: Session = Depends(get_db)):
    try:
        return create_product(db, payload.model_dump())
    except IntegrityError as error:
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail = "Product ID already exists or violates a database constraint"
        ) from error

@app.patch("/products/{product_id}", response_model=ProductRead)
def edit_product(
        product_id : str,
        payload : ProductUpdate,
        db : Session = Depends(get_db)) :

    updates = payload.model_dump(exclude_unset=True)

    if not updates:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "Provide at least one failed to update"
        )

    #Required database fields cannot be explicitly set to null.
    for field in ("name", "category"):
        if field in updates and updates[field] is None:
            raise HTTPException(
                status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail = f"{field} cannot be null"
            )

    product = get_product(db, product_id, updates)

    if product is None:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "Product Not found"
        )
    return product

@app.delete("/products/{product_id}", status_code = status.HTTP_204_NO_CONTENT)
def remove_product(product_id: str, db: Session = Depends(get_db)):
    deleted = delete_product(db, product_id)

    if not deleted:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "Product Not found"
        )

    return Response(status_code = status.HTTP_204_NO_CONTENT)



