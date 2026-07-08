# CI Log Triage Agent

An agentic RAG system that ingests CI run logs (tmt / Fedora CI / pytest style),
indexes them with hybrid retrieval, and answers triage questions like
*"which test failed in run-101 and why?"* — with every claim grounded in
cited log chunks.

## Architecture

```
logs ──> ingest (per-test sections) ──> chunk (provenance-preserving)
                                          │
                              HybridIndex (BM25 + vector, RRF fusion)
                                          │
            LangGraph agent:  retrieve ─> triage ─> [refine ─> retrieve]* ─> answer
                                          │
                        FastAPI service  (/ingest, /query, /health)
```

Design decisions:
- **Hybrid retrieval (BM25 + vector, RRF-fused)** — log triage needs exact-match
  on test names and error codes *and* semantic match on paraphrased questions.
- **Chunking within test sections** — every chunk carries run_id + test_name
  provenance, so citations are unambiguous.
- **Bounded self-correction loop** — if triage confidence < 0.6 the agent
  rewrites its query and re-retrieves, capped at 3 iterations (guards against
  agent loops).
- **Groundedness enforcement** — cited chunk_ids are validated against what was
  actually retrieved; hallucinated citations are dropped.
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
retrieval_hit@6=1.00  triage_accuracy=1.00  groundedness=1.00  (n=6, gate=0.66)
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
PYTHONPATH=src python -m pytest tests/     # unit tests
python evals/run_evals.py                  # eval gate (offline, mock LLM)
PYTHONPATH=src uvicorn ci_triage.api:app   # serve on :8000
curl -X POST localhost:8000/ingest -H 'Content-Type: application/json' \
     -d '{"directory":"sample_logs"}'
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
     -d '{"question":"why is DNS timing out?"}'
```

## Roadmap
- [ ] Ingest real Fedora CI / Testing Farm artifacts
- [ ] API embeddings + Chroma persistent store
- [ ] Langfuse trajectory tracing, per-run cost/latency attribution
- [ ] Grow eval set to 30–50 historically diagnosed failures
- [ ] Cross-run flake detection (same test failing intermittently)
