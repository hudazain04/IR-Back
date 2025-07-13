import ir_datasets
from sqlalchemy.orm import Session
from repositories import qrels_repo
from models.qrels import Qrel

from database import SessionLocal

from fastapi import APIRouter

router = APIRouter()

def load_qrels_to_db(dataset_name , db):
    dataset = ir_datasets.load(dataset_name)

    count = 0
    for qrel in dataset.qrels_iter():

        qrel = Qrel(
             query_id=qrel.query_id,
            doc_id=qrel.doc_id,
            relevance=qrel.relevance,
            source=dataset_name
        )
        qrels_repo.upsert_qrel(
            db,
            qrel
        )
        count += 1
        if count % 50 == 0:
            qrels_repo.commit(db)

    qrels_repo.commit(db)
    print(f"✅ Inserted {count} qrels into the database.")
    return {"status": "success", "message":f"✅ Inserted {count} qrels into the database."}


# if __name__ == "__main__":
#     db = SessionLocal()
#     load_qrels_to_db(db)
