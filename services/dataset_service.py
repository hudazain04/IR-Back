import asyncio
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
import ir_datasets
from models.document import Document
from services.processor import TextProcessor
from repositories import document_repo
import time

router = APIRouter()

def load_dataset(dataset_name: str, db: Session):
    try:
        dataset = ir_datasets.load(dataset_name)
    except Exception as e:
        print(f"event: error\ndata: {str(e)}\n\n")
        return "error"

    processor = TextProcessor()
    count = 0
    MAX_DOCS = 500_000
    BATCH_SIZE = 500
    buffer = []

    start_time = time.time()

    for doc in dataset.docs_iter():
        if count >= MAX_DOCS:
            break

        raw = doc.text
        processed = " ".join(processor.normalize(raw))

        buffer.append({
            "doc_id": doc.doc_id,
            "raw_text": raw,
            "processed_text": processed,
            "source": dataset_name
        })

        count += 1

        if len(buffer) >= BATCH_SIZE:
            batch_start = time.time()
            document_repo.bulk_upsert_documents(db, buffer)
            batch_end = time.time()
            print(f"[Batch Commit] ✅ Committed {len(buffer)} docs | Total so far: {count} "
                  f"| Progress: {count/MAX_DOCS:.2%} | Batch time: {batch_end - batch_start:.2f}s")
            buffer.clear()

    if buffer:
        batch_start = time.time()
        document_repo.bulk_upsert_documents(db, buffer)
        batch_end = time.time()
        print(f"[Final Commit] ✅ Committed {len(buffer)} docs | Total: {count} "
              f"| Final batch time: {batch_end - batch_start:.2f}s")

    total_time = time.time() - start_time
    print(f"[Done] 🎉 Finished loading {count} documents from {dataset_name} in {total_time:.2f} seconds")

    return f"event: done\ndata: Finished loading {count} documents\n\n"
