from pathlib import Path
import pandas as pd
from sqlalchemy import select, func
from src.database.connection import SessionLocal
from src.database.models import Product


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = (PROJECT_ROOT / "data" / "processed" / "products_cleaned.csv")

def optional_text(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def optional_float(value):
    if pd.isna(value):
        return None
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted):
        return None
    return float(converted)


def optional_int(value):
    number = optional_float(value)
    if number is None:
        return None
    if not number.is_integer():
        raise ValueError(f"Stock must be a whole number, got {number}")
    return int(number)

def seed_products():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    required = ["product_id", "name", "category", "brand", "price", "stock", "rating"]

    missing = [
        col for col in required
        if col not in df.columns]

    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    imported = 0
    skipped = 0

    with SessionLocal.begin() as session:
        for row in df.to_dict(orient="records"):
            product_id = optional_text(row["product_id"])
            name = optional_text(row["name"])
            category = optional_text(row["category"])

            if not product_id or not name or not category:
                skipped += 1
                continue

            product = Product(
                product_id=product_id,
                name=name,
                category=category,
                brand=optional_text(row["brand"]),
                price=optional_float(row["price"]),
                stock=optional_int(row["stock"]),
                rating=optional_float(row["rating"])
            )

            # Insert a new product or update an existing
            # product with the same primary key.
            session.merge(product)
            imported += 1

    with SessionLocal() as session:
        total = session.scalar(select(func.count()).select_from(Product))

    print("Import completed.")
    print("Rows processed:", imported)
    print("Rows skipped:", skipped)
    print("Total products in database:", total)

if __name__ == "__main__":
    seed_products()