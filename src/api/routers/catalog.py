from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from src.api.dependencies import get_db
from src.database.models import Product

router = APIRouter(
    prefix = "/catalog",
    tags = ["Catalog & Statistics"]
)

@router.get("/summary")
def get_catalog_summary(db: Session = Depends(get_db)):

    total_products = db.scalar(
        select(func.count()).select_from(Product)
    ) or 0

    total_categories = db.scalar(
        select(func.count(func.distinct(Product.category)))
    ) or 0

    total_brands = db.scalar(
        select(func.count(func.distinct(Product.brand)))
    ) or 0

    total_stock = db.scalar(
        select(func.coalesce(func.sum(Product.stock),0))
    ) or 0

    available_products = db.scalar(
        select(func.count()).select_from(Product).where(Product.stock > 0)
    ) or 0

    out_of_stock_products = db.scalar(
        select(func.count()).select_from(Product).where(Product.stock == 0)
    ) or 0

    unknown_stock_products = db.scalar(
        select(func.count()).select_from(Product).where(Product.stock.is_(None))
    ) or 0


    min_price = db.scalar(
        select(func.min(Product.price)))

    max_price = db.scalar(
        select(func.max(Product.price)))

    average_price = db.scalar(
        select(func.avg(Product.price)))

    inventory_value = db.scalar(
        select(func.coalesce(func.sum(Product.price * Product.stock),0))
    ) or 0

    return {
        "total_products": total_products,
        "total_categories": total_categories,
        "total_brands": total_brands,
        "total_stock_units": total_stock,
        "available_products": available_products,
        "out_of_stock_products": out_of_stock_products,
        "unknown_stock_products" : unknown_stock_products,
        "min_price": min_price,
        "max_price": max_price,
        "average_price": average_price,
        "estimated_inventory_value" : inventory_value
    }


@router.get("/count")
def get_product_count(
        category : str | None = None,
        brand : str | None = None,
        db: Session = Depends(get_db)):

    statement = select(func.count()).select_from(Product)

    if category:
        statement = statement.where(Product.category.ilike(f"%{category}%"))

    if brand:
        statement = statement.where(Product.brand.ilike(f"%{brand}%"))

    total = db.scalar(statement) or 0

    return {
        "category_filter" : category,
        "brand_filter" : brand,
        "product_count" : total
    }

@router.get("/categories")
def get_categories_counts(db: Session = Depends(get_db)):

    statement = (select(Product.category, func.count(Product.product_id).label("product_count"))
    .group_by(Product.category)
    .order_by(func.count(Product.product_id).desc()))

    rows = db.execute(statement).all()
    items = [
        {
            "category" : category,
            "product_count" : count
        } for category, count in rows
    ]

    return {
        "total_categories" : len(items),
        "items" : items
    }

@router.get("/brands")
def get_brand_counts(db: Session = Depends(get_db)):

    statement = (select(Product.brand, func.count(Product.product_id).label("product_count"))
                 .where(Product.brand.is_not(None),
                        Product.brand != "")
                 .group_by(Product.brand)
                 .order_by(func.count(Product.product_id).desc()))

    rows = db.execute(statement).all()

    items = [
        {
            "brand": brand,
            "product_count" : count
        } for brand, count in rows
    ]

    return {
        "total_brands" : len(items),
        "items" : items
    }
