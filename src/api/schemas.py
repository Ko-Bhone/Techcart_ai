from pydantic import BaseModel, ConfigDict, Field


class ProductCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    product_id : str = Field(min_length=1, max_length=100)
    name : str = Field(min_length=1, max_length=500)
    category : str = Field(min_length=1, max_length=120)

    brand : str | None = Field(default= None, max_length=200)
    price : float| None = Field(default=None, ge=0)
    stock : int | None = Field(default=None, ge=0)
    rating : float | None = Field(default=None, ge=0, le=5)


class ProductUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str | None = Field(default=None, min_length=1, max_length=500)
    category: str | None = Field(default=None, min_length=1, max_length=120)

    brand: str | None = Field(default=None, max_length=200)
    price: float | None = Field(default=None, ge=0)
    stock : int | None = Field(default=None, ge=0)
    rating : float | None = Field(default=None, ge=0, le=5)


class ProductRead(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    product_id : str
    name : str
    category : str

    brand : str | None = None
    price : float | None = None
    stock : int | None = None
    rating : float | None = None

class MLPredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    brand : str | None = Field(default=None, max_length=200)
    price : float | None = Field(default=None, ge=0)
    stock : int | None = Field(default=None, ge=0)
    rating : float | None = Field(default=None, ge=0, le=5)

class MLPredictionResponse(BaseModel):

    model_name : str
    prediction_category : str
    probabilities : dict[str, float] | None = None

