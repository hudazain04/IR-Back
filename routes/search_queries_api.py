from fastapi import APIRouter , Depends
from database import get_db
from sqlalchemy.orm import Session
from database import Base, engine
from services.search_queries import run_search_for_all_algorithms
from fastapi.responses import StreamingResponse

router = APIRouter()


@router.post("/search_queries/{datasetName}")
def search_queries_api(dataset_name: str, db: Session = Depends(get_db)):
    return  run_search_for_all_algorithms(dataset_name),
     