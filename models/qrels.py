from sqlalchemy import Column, String, Integer
from database import Base

class Qrel(Base):
    __tablename__ = "qrels"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    query_id = Column(String(255), index=True)
    doc_id = Column(String(255), index=True)
    relevance = Column(Integer, default=1)
    source = Column(String(100), nullable=False)
