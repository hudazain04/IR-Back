
from fastapi import APIRouter, Depends , Query
from sqlalchemy.orm import Session
from services.search_service import SearchService
from database import get_db

router = APIRouter()

search_service = SearchService()

@router.post("/search")
def search(
    query: str = Query(..., description="Search query text"),
    algorithm: str = Query("vsm", regex="^(vsm|word2vec|hybrid|bm25)$", description="Search algorithm to use"),
    top_k: int = Query(5, ge=1, le=50, description="Number of top results to return"),
    dataset : str = Query(..., description="name of dataset"),
    with_index: bool = False,
    with_additional : bool =False
):
    return search_service.search(query, algorithm , dataset ,  top_k , with_index,with_additional)
