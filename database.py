# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


# DATABASE_URL = "sqlite:///data/ir_data.db"

# engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SQLALCHEMY_DATABASE_URL = "sqlite:///D:/huda1/IR/IR-Project/data/ir_data.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


