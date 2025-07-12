from fastapi import APIRouter , Depends
from database import get_db
from sqlalchemy.orm import Session
from database import Base, engine
from services.load_queries import load_queries_to_db
from fastapi.responses import StreamingResponse

router = APIRouter()


@router.post("/load_queries/{datasetName}")
def load_queries_api(dataset_name: str, db: Session = Depends(get_db)):
    return  load_queries_to_db(dataset_name),
     