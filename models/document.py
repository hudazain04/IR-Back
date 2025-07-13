from sqlalchemy import Column, String, Text
from database import Base 

class Document(Base):
    __tablename__ = 'documents'

    doc_id = Column(String(255), primary_key=True, index=True)
    raw_text = Column(Text)
    processed_text = Column(Text)
    source = Column(String(100))