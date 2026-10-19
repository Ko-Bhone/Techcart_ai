from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from src.database.models import Product

def create_product(session: Session, product_data:dict) -> Product:

    product = Product(**product_data)
    session.add(product)

    try:
        session.commit()
        session.refresh(product)
    except IntegrityError:
        session.rollback()
        raise
    return product

def get_product(session: Session, product_id: str) -> Product | None:
    return session.get(Product, str(product_id))

def list_products(
        session: Session,
        category : str | None = None,
        brand: str | None = None,
        limit : int = 100,
        offset : int = 0) -> list[Product]:

    statement = select(Product)
    if category:
        statement = statement.where(Product.category.ilike(f"%{category}%"))

    if brand:
        statement = statement.where(Product.brand.ilike(f"%{brand}%"))

    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    statement = (statement.order_by(Product.product_id).limit(limit).offset(offset))

    return list(session.scalars(statement).all())

def update_product(session: Session,
                   product_id: str,
                   updates : dict) -> Product | None:

    product = get_product(session, product_id)
    if product is None:
        return None

    allowed_fields = {
        "name",
        "category",
        "brand",
        "price",
        "stock",
        "rating"
    }

    invalid_fields = set(updates) - allowed_fields

    if invalid_fields:
        raise ValueError(f"Fields cannot be updated: {invalid_fields}")

    for field, value in updates.items():
        setattr(product, field, value)

    try:
        session.commit()
        session.refresh(product)

    except Exception:
        session.rollback()
        raise

    return product

def delete_product(session: Session, product_id: str) -> bool:
    product = get_product(session, product_id)

    if product is None:
        return False

    try:
        session.delete(product)
        session.commit()

    except Exception:
        session.rollback()
        raise

    return True