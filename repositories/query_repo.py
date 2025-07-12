from sqlalchemy.orm import Session
from models.query import Query
from typing import List

def upsert_query(db: Session, query: Query):
    db.merge(query)

def get_queries_by_source(db: Session, source: str) -> List[Query]:
    return db.query(Query).filter(Query.source == source).all()

def commit(db: Session):
    db.commit()
    