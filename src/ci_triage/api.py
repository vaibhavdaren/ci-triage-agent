"""FastAPI service around the triage agent.

    uvicorn ci_triage.api:app --reload

POST /ingest  {"directory": "sample_logs"}   -> builds the index
POST /query   {"question": "which test failed in run 42?"}
GET  /health
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import ask
from .chunk import chunk_runs
from .ingest import parse_directory
from .retrieve import HybridIndex

app = FastAPI(title="CI Log Triage Agent", version="0.1.0")
_index: HybridIndex | None = None


class IngestRequest(BaseModel):
    directory: str


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": len(_index.chunks) if _index else 0}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    global _index
    runs = parse_directory(req.directory)
    if not runs:
        raise HTTPException(404, f"no .log/.txt files under {req.directory}")
    chunks = chunk_runs(runs)
    _index = HybridIndex()
    _index.build(chunks)
    return {"runs": len(runs), "chunks": len(chunks)}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    if _index is None:
        raise HTTPException(409, "call /ingest first")
    return ask(_index, req.question)
