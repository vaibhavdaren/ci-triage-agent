# CI Log Triage Agent

A multi-agent RAG system that ingests CI run logs (tmt / Fedora CI / pytest style),
indexes them with hybrid retrieval, and answers triage questions like
*"which test failed in run-101 and why?"* — with every claim grounded in
cited log chunks and enriched via MCP tool calls.

## Architecture

```
logs ──> ingest (per-test sections) ──> chunk (provenance-preserving)
                                          │
                              HybridIndex (BM25 + vector, RRF fusion)
                                          │
  LangGraph, 4 collaborating agents:
    Retriever ─> Diagnosis ─> Critic ──accept──> Tool (MCP) ─> answer
                     ▲             │
                     └──refine─────┘   (bounded, MAX_ITERATIONS=3)
                                          │
             FastAPI service  (/ingest, /query, /tools, /health)
                                          │
             MCP tool server (stdio)  — fetch_build_artifact, lookup_test_owner
```

Design decisions:
- **Hybrid retrieval (BM25 + vector, RRF-fused)** — log triage needs exact-match
  on test names and error codes *and* semantic match on paraphrased questions.
- **Chunking within test sections** — every chunk carries run_id + test_name
  provenance, so citations are unambiguous.
- **Four specialized, collaborating agents** (`src/ci_triage/agents.py`) — a
  Retriever, a Diagnosis Agent (LLM), a rule-based Critic Agent, and a Tool
  Agent. The Diagnosis Agent proposes a grounded answer; the Critic
  independently verifies it and either accepts or sends the Retriever a
  refined query — a bounded collaboration loop, not an open one
  (`MAX_ITERATIONS = 3` guards against the classic agent failure mode of
  looping forever).
- **Two-layer hallucination guard** — the Critic checks (1) every cited
  `chunk_id` was actually retrieved, dropping any that weren't, and (2) the
  root-cause text lexically overlaps with the text of what it cites, catching
  claims that aren't actually supported by the evidence. This is deliberately
  *not* another LLM call grading the first one's output — grounding is
  checked programmatically.
- **MCP-based tool integration** (`src/ci_triage/mcp/`) — a local MCP server
  exposes tools (`fetch_build_artifact`, `lookup_test_owner`) via a
  `@mcp.tool()` decorator; the Tool Agent discovers and calls them through a
  standard MCP client after the Critic accepts an answer. Adding a new tool
  is one function + one decorator — no other code changes. Inputs like
  `run_id` are validated before use as lookup keys.
- **Persistent, concurrency-safe index** — `HybridIndex.save`/`load` persist
  the fitted BM25 + TF-IDF state to disk so a restart doesn't lose an ingested
  corpus; a lock guards the shared index against concurrent requests.
- **Pluggable backends** — TF-IDF embedder and a deterministic mock LLM run
  fully offline; set `ANTHROPIC_API_KEY` for Claude-backed diagnosis, or swap
  in an API embedder / Chroma / FAISS via the `Embedder` protocol.

## Evaluation as a regression gate

`evals/eval_set.jsonl` holds triage cases with known root causes. The harness
measures **retrieval_hit@6**, **triage_accuracy**, and **groundedness**, and
exits non-zero below threshold — wire it into CI (or the Docker build) so a
regressed change never ships.

```
$ python evals/run_evals.py
retrieval_hit@6=1.00  triage_accuracy=0.95  groundedness=1.00  (n=20, gate=0.66)
gate passed
```

The eval loop caught three real bugs during development: pytest
`=== FAILURES ===` banners being parsed as test boundaries (orphaning
tracebacks from their failing test), chunk-id `:` separators colliding with
pytest `::` names, and hyphenated test names being invisible to lexical
search. Each fix was verified by re-running the gate.

## Quickstart

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests/     # unit tests (incl. MCP + multi-agent loop)
python evals/run_evals.py                  # eval gate (offline, mock LLM + local MCP server)
PYTHONPATH=src uvicorn ci_triage.api:app   # serve on :8000
curl -X POST localhost:8000/ingest -H 'Content-Type: application/json' \
     -d '{"directory":"sample_logs"}'
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
     -d '{"question":"why is DNS timing out?"}'
curl localhost:8000/tools                  # MCP tools available to the Tool Agent
```

The index persists to `CI_TRIAGE_INDEX_PATH` (default `.data/index.pkl`) after
every `/ingest` and reloads on startup — restarting the service doesn't lose
an already-ingested corpus.

## Roadmap
- [ ] Ingest real Fedora CI / Testing Farm artifacts
- [ ] API embeddings + Chroma persistent store
- [ ] Langfuse trajectory tracing, per-run cost/latency attribution beyond the
      basic `_latency_ms`/structured logging already in `api.py`
- [ ] Grow eval set toward 30–50 historically diagnosed failures (currently 20)
- [ ] Cross-run flake detection (same test failing intermittently)
- [ ] MCP tools backed by real services (GitHub Issues, artifact storage)
      instead of the local mock data in `src/ci_triage/mcp/data/`
