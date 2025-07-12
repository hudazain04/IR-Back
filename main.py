from fastapi import FastAPI 
from routes import load_dataset_api
from routes import tfidf_api
from routes import word2vec_api
from routes import inverted_index_api
from routes import search_api
from routes import bm25
from routes import search_queries_api
from routes import build_faiss_index_store
from routes import load_qrels_api
from routes import load_queries_api
from database import Base, engine
from models.document import Document
from models.query import Query
from models.qrels import Qrel
from models.search_result import SearchResult
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
origins = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
    # Add your production domain if needed
]


Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for all origins (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)
app.include_router(load_dataset_api.router)
app.include_router(tfidf_api.router)
app.include_router(word2vec_api.router)
app.include_router(inverted_index_api.router)
app.include_router(search_api.router)
app.include_router(bm25.router)
app.include_router(build_faiss_index_store.router)
app.include_router(search_queries_api.router)
app.include_router(load_qrels_api.router)
app.include_router(load_queries_api.router)




@app.get("/")
def root() :
    return {"Fast api server for IR project"}