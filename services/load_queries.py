import ir_datasets
from sqlalchemy.orm import Session
from repositories import query_repo
from models.query import Query
from fastapi import APIRouter

router = APIRouter()


def load_queries_to_db(dataset_name , db : Session):
    print("Starting query load...")

    dataset = ir_datasets.load(dataset_name)
    count = 0

    for q in dataset.queries_iter():
        query_id = f"{dataset}:{q.query_id}"
        raw = q.text
        query = Query(
            query_id=q.query_id,
            raw_text=raw,
            source=dataset_name,
        )

        query_repo.upsert_query(db, query)
        count += 1

        if count % 50 == 0:
            print(f"Inserted {count} queries so far...")
            query_repo.commit(db)

    query_repo.commit(db)
    print(f"✅ All {count} queries inserted into DB.")
    return {"status": "success", "message":f"✅ All {count} queries inserted into DB."}
