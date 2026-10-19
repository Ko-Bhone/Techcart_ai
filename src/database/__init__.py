from src.database.connection import engine
from src.database.models import Base, Product

def init_db():
    Base.metadata.create_all(engine)
    print(f"Database Tables Created Successfully")

if __name__ == "__main__":
    init_db()