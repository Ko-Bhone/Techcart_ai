from fastapi import (APIRouter, Depends, HTTPException, Query, Response, status)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from src.api.dependencies import get_db
from src.api.schemas import ProductCreate, ProductUpdate, ProductRead
from src.database.crud import create_product, update_product, get_product, list_products, delete_product

router = APIRouter(
    prefix="/products",
    tags = ["Products"]
)

#Product list ကို category/brand အလိုက် ရှာမယ်။
@router.get("/", response_model=list[ProductRead])
def get_products(
        category: str | None = None,
        brand : str | None = None,
        limit : int = Query(default = 100, ge=1, le=500),
        offset : int = Query(default = 0, ge=0),
        db : Session = Depends(get_db)):

    return list_products(
        db,
        category = category,
        brand = brand,
        limit = limit,
        offset = offset
    )

@router.get("/{product_id", response_model = ProductRead)
def get_product_by_id(
        product_id : str,
        db : Session = Depends(get_db)):

    product = get_product(db, product_id)

    if product is None:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND,
                            detail = "Product not found")
    return product

@router.post("/",
             response_model=ProductRead,
             status_code = status.HTTP_201_CREATED)
def add_product(
        payload : ProductCreate,
        db : Session = Depends(get_db)):
    try:
        return create_product(db, payload.model_dump())

    except IntegrityError as error:
        raise HTTPException(status_code = status.HTTP_409_CONFLICT,
                            detail = "Product ID already exists or violates a database constraint") from error

@router.patch("/{product_id}", response_model=ProductRead)
def edit_product(
        product_id : str,
        payload : ProductUpdate,
        db : Session = Depends(get_db)):

    updates = payload.model_dump(exclude_unset=True)

    if not updates:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST,
                            detail = "Provide at least one field to update")

    for field in ("name","category"):
        if field in updates and updates[field] is None:
            raise HTTPException(status_code = 422, detail = f"{field} cannot ne null")

    product = update_product(db, product_id, updates)

    if product is None:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND,
                            detail = "Product not found")
    return product


@router.delete("/{product_id}", status_code = status.HTTP_204_NO_CONTENT)
def delete_product(product_id : str, db: Session = Depends(get_db)):

    deleted = delete_product(db, product_id)

    if not deleted:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "Product not found"
        )

    return Response(status_code = status.HTTP_204_NO_CONTENT)