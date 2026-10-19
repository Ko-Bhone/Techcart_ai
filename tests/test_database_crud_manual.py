import uuid

from src.database.connection import SessionLocal
from src.database.crud import (
    create_product,
    get_product,
    list_products,
    update_product,
    delete_product,
)


def test_product_crud():
    demo_id = f"DEMO-{uuid.uuid4().hex[:8]}"

    with SessionLocal() as session:

        # 1. CREATE
        created = create_product(
            session,
            {
                "product_id": demo_id,
                "name": "Demo NVMe SSD",
                "category": "SSD",
                "brand": "TechCart",
                "price": 89.99,
                "stock": 12,
                "rating": 4.5,
            },
        )

        try:
            assert created.product_id == demo_id
            assert created.name == "Demo NVMe SSD"
            print("PASS: CREATE")

            # 2. READ
            product = get_product(session, demo_id)

            assert product is not None
            assert product.name == "Demo NVMe SSD"
            assert product.price == 89.99
            assert product.stock == 12
            print("PASS: READ")

            # 3. UPDATE
            updated = update_product(
                session,
                demo_id,
                {
                    "price": 79.99,
                    "stock": 20,
                },
            )

            assert updated is not None
            assert updated.price == 79.99
            assert updated.stock == 20
            print("PASS: UPDATE")

            # 4. LIST
            ssd_products = list_products(
                session,
                category="SSD",
                limit=500,
            )

            assert any(
                p.product_id == demo_id
                for p in ssd_products
            )
            print("PASS: LIST")

        finally:
            # Remove the temporary test product
            if get_product(session, demo_id) is not None:
                delete_product(session, demo_id)

        # 5. VERIFY DELETION
        assert get_product(session, demo_id) is None
        print("PASS: DELETE")
        print("PASS: VERIFY DELETION")