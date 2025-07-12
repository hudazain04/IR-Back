from fastapi import APIRouter , Depends
from database import get_db
from sqlalchemy.orm import Session
from database import Base, engine
from services.load_qrels import load_qrels_to_db
from fastapi.responses import StreamingResponse

router = APIRouter()


@router.post("/load_qrels/{datasetName}")
def load_qrels_api(dataset_name: str, db: Session = Depends(get_db)):
    return  load_qrels_to_db(dataset_name),
     