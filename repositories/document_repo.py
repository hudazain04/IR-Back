from sqlalchemy.orm import Session
from models.document import Document
from typing import List
from sqlalchemy.dialects.mysql import insert

def upsert_document(db: Session, doc: Document):
    db.merge(doc)



def bulk_upsert_documents(db: Session, documents: list[dict]):
    if not documents:
        return

    stmt = insert(Document).values(documents)

    update_stmt = {
        "raw_text": stmt.inserted.raw_text,
        "processed_text": stmt.inserted.processed_text,
        "source": stmt.inserted.source,
    }

    upsert_stmt = stmt.on_duplicate_key_update(**update_stmt)

    db.execute(upsert_stmt)
    db.commit()


def get_documents_by_source(db: Session, source: str) -> List[Document]:
    return db.query(Document).filter(Document.source == source).all()

def commit(db: Session):
    db.commit()