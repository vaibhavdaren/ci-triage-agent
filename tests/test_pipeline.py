import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ci_triage.chunk import chunk_runs
from ci_triage.ingest import parse_directory
from ci_triage.retrieve import HybridIndex

ROOT = Path(__file__).resolve().parents[1]


def _index():
    runs = parse_directory(ROOT / "sample_logs")
    idx = HybridIndex()
    idx.build(chunk_runs(runs))
    return runs, idx


def test_parsing_finds_failures():
    runs, _ = _index()
    failed = {s.test_name for r in runs for s in r.failed_sections}
    assert "/tests/network/dns-resolution" in failed
    assert any("test_koji_fetch" in f for f in failed)


def test_chunks_carry_provenance():
    runs, _ = _index()
    chunks = chunk_runs(runs)
    assert all(c.run_id and c.test_name for c in chunks)


def test_hybrid_retrieval_exact_match():
    _, idx = _index()
    top = idx.search("dns.exception.Timeout resolver", k=3)
    assert any("dns-resolution" in c.test_name for c in top)


def test_hybrid_retrieval_semantic():
    _, idx = _index()
    top = idx.search("workers stuck waiting on a file lock", k=3, failed_only=True)
    assert any("parallel-suite" in c.test_name for c in top)
