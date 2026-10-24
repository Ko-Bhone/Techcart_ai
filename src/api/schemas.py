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
    predicted_category : str
    probabilities : dict[str, float] | None = None

class NLPTextPredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    product_name : str = Field(min_length=1, max_length=1000)

class NLPTextPredictionResponse(BaseModel):
    model_name : str
    product_name : str
    cleaned_text : str
    predicted_category : str
    probabilities : dict[str, float] | None = None

class RAGQueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question : str = Field(min_length=1, max_length=2000)
    top_k : int = Field(default=4, ge=1, le=10)


class RAGSource(BaseModel):

    product_id : str | None = None
    name : str | None = None
    category : str | None = None
    brand : str | None = None

    price : float | None = None
    stock : int | None = None
    rating : float | None = None

    similarity : float
    document : str


class RAGSearchResponse(BaseModel):
    question : str
    model_name : str = "catalog-vector-search"
    results : list[RAGSource]

class RAGAnswerResponse(BaseModel):
    question : str
    answer : str
    model_name : str
    sources : list[RAGSource] = Field(default_factory = list)



