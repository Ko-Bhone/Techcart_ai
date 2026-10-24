TechCart AI — AI-Powered IT Hardware E-Commerce Platform
TechCart AI is a portfolio project that combines an IT-hardware product catalog and database-backed REST API with machine-learning classification, product-name NLP, and retrieval-augmented generation (RAG).
Project status: In development. The API and automated tests are being stabilized. The endpoints listed below reflect the current project structure; run the tests in your local environment before treating a feature as verified.

Features
- Product management: Create, list, retrieve, update, and delete products using a FastAPI REST API.
- Catalog analytics: Summary metrics, product counts, category lists, and brand lists.
- Classical ML: Predict a product category from its brand, price, stock, and rating using a saved scikit-learn pipeline.
- NLP classification: Predict a category from a product name using a saved TF-IDF text-classification pipeline.
- RAG retrieval: Search for relevant catalog items using saved embeddings/vector-store artifacts.
- RAG question answering: Generate an answer from retrieved product context using a local Ollama model.
- PostgreSQL integration: Store and manage the product catalog with SQLAlchemy.
- Experiment notebooks: EDA, preprocessing, model training, NLP, RAG, and database exploration notebooks.
Technology Stack
- Python 3.12+
- FastAPI, Uvicorn, Pydantic
- PostgreSQL, SQLAlchemy, Psycopg
- pandas, NumPy, scikit-learn, joblib
- PyTorch
- TF-IDF and Sentence Transformers
- Ollama (configured for qwen2.5:3b in the current RAG service)
- pytest and FastAPI TestClient
Architecture





```mermaid
flowchart TD
    Client[Client / Swagger UI] --> API[FastAPI application]
    API --> CRUD[Product CRUD routes]
    CRUD --> DB[(PostgreSQL)]
    API --> Catalog[Catalog analytics]
    Catalog --> DB
    API --> ML[scikit-learn category prediction]
    ML --> MLArtifacts[Saved ML pipeline]
    API --> NLP[Product-name NLP prediction]
    NLP --> NLPArtifacts[Saved TF-IDF pipeline]
    API --> RAGSearch[RAG catalog retrieval]
    RAGSearch --> VectorStore[Saved vector store and embeddings]
    API --> RAGAsk[RAG question answering]
    RAGAsk --> Ollama[Local Ollama model]
```
Repository Structure
Techcart_ai/
├── data/
│   ├── raw/                 # Source product and interaction data
│   ├── processed/           # Cleaned data and prepared splits
│   └── splits/              # Train / validation / test datasets
├── knowledge_base/          # RAG knowledge-base resources
├── models/                  # Saved ML, NLP, PyTorch, and RAG artifacts
├── notebooks/               # EDA, training, NLP, RAG, and database notebooks
├── src/
│   ├── api/                 # FastAPI application, schemas, dependencies, routers
│   ├── data/                # Data loading and preprocessing helpers
│   ├── database/            # SQLAlchemy connection, models, CRUD, seeding
│   ├── dl/                  # PyTorch dataset, model, training, inference
│   ├── ml/                  # Classical ML training, evaluation, inference
│   ├── nlp/                 # Text preprocessing, embedding, search, sentiment
│   └── rag/                 # RAG service
└── tests/                   # Automated and manual tests
Getting Started
1. Clone the repository
git clone https://github.com/Ko-Bhone/Techcart_ai.git
cd Techcart_ai
2. Create and activate a virtual environment
Python 3.12 is the development environment currently used for this project.

macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
Windows PowerShell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
3. Install Python dependencies
python -m pip install -r requirements.txt
PyTorch installation can vary by operating system and hardware. If pip selects an unsuitable build for your machine, use the official PyTorch installation selector and then install the remaining dependencies from requirements.txt.
4. Configure PostgreSQL
Create a PostgreSQL database for the project, for example techcart_ai, then create a .env file in the repository root. Do not commit .env or share its database password.

Example .env entry:
DATABASE_URL=postgresql+psycopg://YOUR_DB_USER:YOUR_DB_PASSWORD@localhost:5432/techcart_ai
Replace the placeholders with your own local PostgreSQL credentials. Ensure the database schema exists and the catalog has been seeded before calling database-backed endpoints. Use the existing database model and seeding code under src/database/ to initialize your local database.
5. Check the saved model artifacts
The current API loads model and RAG artifacts from the models/ directory. Make sure the artifacts produced by the notebooks are present before starting the relevant endpoints, including the files used for:
- Classical ML classification and its metadata
- Product-name TF-IDF classification and metadata
- PyTorch category classification and its preprocessor/label encoder
- Product catalog vector store and RAG metadata
These artifacts are not generated merely by installing requirements.txt; run the relevant notebook/training workflow if an artifact is missing. Keep any required artifacts available to the API when cloning or deploying the project.
6. Start the API
From the repository root, run:
uvicorn src.api.main:app --reload
Open the interactive API documentation at:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- Root endpoint: http://127.0.0.1:8000/
- Health endpoint: http://127.0.0.1:8000/health
API Endpoints
Method	Endpoint	Purpose
GET	/	API welcome response
GET	/health	API/database health check
GET	/products/	List products, with supported filters/pagination
POST	/products/	Create a product
GET	/products/{product_id}	Retrieve one product
PATCH	/products/{product_id}	Update product fields
DELETE	/products/{product_id}	Delete a product
GET	/catalog/summary	Catalog summary metrics
GET	/catalog/count	Count products, optionally filtered
GET	/catalog/categories	List/count product categories
GET	/catalog/brands	List/count brands
POST	/ml/predict-category	Predict category from structured product features
POST	/nlp/predict-category	Predict category from product-name text
POST	/rag/search	Retrieve relevant catalog items without an LLM call
POST	/rag/ask	Answer a question using retrieved catalog context and an LLM


Check /docs for the exact request and response schemas in the currently running version.
RAG / Ollama Setup
The RAG question-answering service is configured to use a local Ollama model (qwen2.5:3b). Install and start Ollama, then pull the model:
ollama pull qwen2.5:3b
Make sure the Ollama API is reachable at the URL configured in src/rag/service.py (the current local default is http://localhost:11434). The /rag/search endpoint performs retrieval only; /rag/ask also needs the LLM service. If Ollama is unavailable, question-answering requests may fail even when retrieval works.
Run Tests
From the repository root, with the virtual environment activated and database/model prerequisites configured:
python -m pytest --collect-only -q
python -m pytest tests/test_api.py -v
python -m pytest -v
The API tests use the configured database and local model artifacts. Some RAG tests may load an embedding model on first run, which can require an internet connection to download model weights unless the weights are already cached. Do not consider the project fully verified until the relevant tests pass in your own environment.
Notebooks
The notebooks/ directory contains the project workflow, including:
1. EDA and preprocessing
2. Classical ML classification
3. PyTorch product-category classification
4. Product-name NLP classification
5. RAG product Q&A
6. Database CRUD exploration
Run the notebooks in order when rebuilding the data and model artifacts, and verify that each notebook's output paths match the API's expected files under models/.
Security Notes
- Keep database credentials in .env; never commit secrets.
- Do not put real customer or payment information in sample data or logs.
- Review and validate all incoming API data before using the service beyond local development.
- This project is a learning/portfolio application and should not be treated as production-hardened software without further security, reliability, and deployment review.