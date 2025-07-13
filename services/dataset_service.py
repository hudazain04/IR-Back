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
    
    for doc in dataset.docs_iter():
        # doc_id = f"{dataset_name}:{doc.doc_id}"
        raw = doc.text
        processed = " ".join(processor.normalize(raw))

        document = Document(
            doc_id=doc.doc_id,
            raw_text=raw,
            processed_text=processed,
            source=dataset_name,
        )

        document_repo.upsert_document(db, document)
        count += 1


        # Optionally, commit periodically every N docs or every few seconds
        if count % 1000 == 0:
            document_repo.commit(db)
            print(count)

    document_repo.commit(db)
    return f"event: done\ndata: Finished loading {count} documents\n\n"

