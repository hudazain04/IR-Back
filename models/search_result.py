from sqlalchemy import Column, String, Integer, Float , Boolean
from database import Base

class SearchResult(Base):
    __tablename__ = "search_results"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(String(255), index=True)
    doc_id = Column(String(255), index=True)
    rank = Column(Integer)
    score = Column(Float)
    algorithm = Column(String(100))
    with_additional=Column(Boolean)
    dataset = Column(String(100))
