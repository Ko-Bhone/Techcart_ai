from fastapi import FastAPI
from src.api.routers import health, products, catalog, ml


app = FastAPI(
    title = "TechCart AI API",
    description = "Product Management and Catalog Analytic API",
    version = "1.0.0"
)

app.include_router(health.router)
app.include_router(products.router)
app.include_router(catalog.router)
app.include_router(ml.router)