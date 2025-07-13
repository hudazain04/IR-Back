# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


DATABASE_URL = "mysql+pymysql://root@localhost:3306/ir_data"

engine = create_engine(DATABASE_URL, echo=False ,  pool_size=20,      
    max_overflow=30,      
    pool_pre_ping=True,   
    pool_recycle=3600 ) 

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


