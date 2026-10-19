from sqlalchemy import String, Float, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass

class Product(Base):
    __tablename__ = "products"
    product_id : Mapped[str] = mapped_column(String(100), primary_key=True)
    name : Mapped[str] = mapped_column(String(500), nullable=False)
    category : Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    brand : Mapped[float | None] = mapped_column(String(200), nullable=True, index=True)
    price : Mapped[float | None] = mapped_column(Float, nullable=True)
    stock : Mapped[float | None] = mapped_column(Integer, nullable=True)
    rating : Mapped[float | None] = mapped_column(Float, nullable=True)

    def __repr__(self):
        return (
            f"Product (product_id = {self.product_id!r},"
            f"name = {self.name!r},"
            f"category = {self.category!r})"
        )

