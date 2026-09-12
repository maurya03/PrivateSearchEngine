from fastapi import FastAPI, Query
from pydantic import BaseModel
from .engine import SearchEngine

app = FastAPI(title="PrivateSearch API", version="1.0.0")
engine = SearchEngine("search.db")

class SearchRequest(BaseModel):
    q: str
    k: int = 10
    field: str = "all"

@app.get("/health")
def health(): return {"status": "ok", "documents": engine.count()}

@app.post("/search")
def search(req: SearchRequest): return {"query": req.q, "results": engine.search(req.q, req.k, field=req.field)}
