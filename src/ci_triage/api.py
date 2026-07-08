"""FastAPI service around the triage agent.

    uvicorn ci_triage.api:app --reload

POST /ingest  {"directory": "sample_logs"}   -> builds the index
POST /query   {"question": "which test failed in run 42?"}
GET  /tools                                  -> MCP tools available to the ToolAgent
GET  /health

The index is persisted to disk (CI_TRIAGE_INDEX_PATH) after every
/ingest and reloaded on startup, so a restart doesn't lose it. A lock
guards the global index against concurrent ingest/query requests.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import ask
from .chunk import chunk_runs
from .ingest import parse_directory
from .mcp import client as mcp_client
from .retrieve import HybridIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("ci_triage")

INDEX_PATH = Path(os.environ.get("CI_TRIAGE_INDEX_PATH", ".data/index.pkl"))

app = FastAPI(title="CI Log Triage Agent", version="0.1.0")
_index: HybridIndex | None = None
_index_lock = threading.Lock()


@app.on_event("startup")
def _load_persisted_index() -> None:
    global _index
    if INDEX_PATH.exists():
        with _index_lock:
            _index = HybridIndex.load(INDEX_PATH)
        logger.info("loaded persisted index: %d chunks from %s", len(_index.chunks), INDEX_PATH)


class IngestRequest(BaseModel):
    directory: str


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    with _index_lock:
        return {"status": "ok", "indexed_chunks": len(_index.chunks) if _index else 0}


@app.get("/tools")
def tools() -> dict:
    try:
        return {"tools": mcp_client.list_tools()}
    except Exception as exc:
        raise HTTPException(503, f"MCP tool server unavailable: {exc}") from exc


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    global _index
    runs = parse_directory(req.directory)
    if not runs:
        raise HTTPException(404, f"no .log/.txt files under {req.directory}")
    chunks = chunk_runs(runs)
    new_index = HybridIndex()
    new_index.build(chunks)
    with _index_lock:
        _index = new_index
        _index.save(INDEX_PATH)
    logger.info("ingested: runs=%d chunks=%d", len(runs), len(chunks))
    return {"runs": len(runs), "chunks": len(chunks)}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    with _index_lock:
        index = _index
    if index is None:
        raise HTTPException(409, "call /ingest first")
    start = time.perf_counter()
    result = ask(index, req.question)
    latency_ms = (time.perf_counter() - start) * 1000
    result["_latency_ms"] = round(latency_ms, 1)
    logger.info(
        "query: iterations=%d confidence=%.2f latency_ms=%.1f question=%r",
        result.get("_iterations", 0),
        result.get("confidence", 0),
        latency_ms,
        req.question,
    )
    return result
