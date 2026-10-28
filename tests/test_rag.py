from src.rag.service import (_answer_looks_repetitive, _format_product_answer)

def make_source(product_id, name, price, stock, rating,):
    return {
        "similarity": 0.8,
        "document": f"Product: {name}",
        "metadata": {
            "product_id": product_id,
            "name": name,
            "category": "SSD",
            "brand": "Samsung",
            "price": price,
            "stock": stock,
            "rating": rating,
        },
    }


def test_format_product_answer():
    sources = [
        make_source(
            product_id=19,
            name="2TB 990 PRO",
            price=161.14,
            stock=30,
            rating=4.9,
        ),
        make_source(
            product_id=187,
            name="2TB 990 PRO",
            price=176.65,
            stock=16,
            rating=4.7,
        ),
        make_source(
            product_id=7,
            name="2TB 990 PRO",
            price=162.96,
            stock=31,
            rating=4.8,
        )
    ]

    answer = _format_product_answer(sources)

    assert "Product ID: 19" in answer
    assert "Price: 161.14" in answer
    assert "Stock: 30" in answer
    assert "Rating: 4.9/5" in answer

    # The lowest-priced product must be identified correctly.
    assert "ဈေးအနိမ့်ဆုံး" in answer
    assert "Product ID: 19" in answer


def test_format_product_answer_empty_sources():
    answer = _format_product_answer([])

    assert "မတွေ့ရှိပါ" in answer


def test_detect_repetitive_answer():
    answer = (
        "SSD SSD SSD SSD SSD "
        "Samsung product stock rating price")

    assert _answer_looks_repetitive(answer) is True


def test_accept_non_repetitive_answer():
    answer = (
        "Samsung SSD products have different "
        "prices, stock quantities, and ratings."
    )

    assert _answer_looks_repetitive(answer) is False