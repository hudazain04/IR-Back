from sqlalchemy.orm import Session
from models.qrels import Qrel
from typing import List

def upsert_qrel(db: Session, qrel: Qrel):
    db.merge(qrel)

def get_qrels_by_source(db: Session, source: str) -> List[Qrel]:
    return db.query(Qrel).filter(Qrel.source == source).all()

def commit(db: Session):
    db.commit()