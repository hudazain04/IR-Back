from sqlalchemy import Column, String, Text
from database import Base 

class Query(Base):
    __tablename__ = 'queries'

    query_id = Column(String, primary_key=True)
    raw_text = Column(Text)
    processed_text = Column(Text)
    source = Column(String, nullable=False)

