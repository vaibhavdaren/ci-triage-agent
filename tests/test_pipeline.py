import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ci_triage import agents as agents_mod
from ci_triage.agent import ask
from ci_triage.agents import CriticAgent
from ci_triage.chunk import Chunk, chunk_runs
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


def test_index_persistence_roundtrip(tmp_path):
    _, idx = _index()
    path = tmp_path / "index.pkl"
    idx.save(path)
    loaded = HybridIndex.load(path)
    assert len(loaded.chunks) == len(idx.chunks)
    top = loaded.search("dns.exception.Timeout resolver", k=3)
    assert any("dns-resolution" in c.test_name for c in top)


def test_critic_rejects_unretrieved_citation():
    chunks = [Chunk("run-1|t|0", "run-1", "t", "some log text about a timeout", failed=True)]
    answer = {
        "failing_test": "t",
        "root_cause": "some log text about a timeout",
        "evidence": ["run-1|t|999"],  # never retrieved
        "confidence": 0.9,
    }
    result = CriticAgent().run(answer, chunks)
    assert result["accepted"] is False
    assert result["answer"]["evidence"] == []
    assert "citation" in result["critic_feedback"]


def test_critic_rejects_ungrounded_root_cause():
    chunks = [Chunk("run-1|t|0", "run-1", "t", "resolver timed out waiting for DNS response", failed=True)]
    answer = {
        "failing_test": "t",
        "root_cause": "totally unrelated explanation about disk corruption and kernel panics",
        "evidence": ["run-1|t|0"],
        "confidence": 0.9,
    }
    result = CriticAgent().run(answer, chunks)
    assert result["accepted"] is False
    assert "overlap" in result["critic_feedback"]


def test_critic_accepts_grounded_answer():
    chunks = [Chunk("run-1|t|0", "run-1", "t", "resolver timed out waiting for DNS response", failed=True)]
    answer = {
        "failing_test": "t",
        "root_cause": "resolver timed out waiting for DNS response",
        "evidence": ["run-1|t|0"],
        "confidence": 0.9,
    }
    result = CriticAgent().run(answer, chunks)
    assert result["accepted"] is True
    assert result["answer"]["evidence"] == ["run-1|t|0"]


def test_multi_agent_refine_loop_then_accepts(monkeypatch):
    """First diagnosis is ungrounded and must be rejected by the Critic;
    the resulting refined_query drives a second retrieve/diagnose pass
    that the Critic accepts. Exercises the real Retriever-Diagnosis-Critic
    collaboration loop, not just each agent in isolation."""
    calls = {"n": 0}

    def fake_llm(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {
                    "failing_test": "unknown",
                    "root_cause": "made up explanation unrelated to any retrieved log text",
                    "evidence": ["not-a-real-chunk-id"],
                    "confidence": 0.9,
                    "refined_query": "dns timeout resolver",
                }
            )
        ids = re.findall(r"\[chunk_id=([^\]]+)\]", prompt)
        top = ids[0] if ids else "unknown"
        test = top.split("|")[1] if "|" in top else "unknown"
        m = re.search(r"\[chunk_id=" + re.escape(top) + r"\][^\n]*\n(.*?)(?=\n\[chunk_id=|\Z)", prompt, re.S)
        snippet = " ".join((m.group(1) if m else "").split())[:200]
        return json.dumps(
            {
                "failing_test": test,
                "root_cause": snippet,
                "evidence": [top],
                "confidence": 0.9,
                "refined_query": "",
            }
        )

    monkeypatch.setattr(agents_mod, "_get_llm", lambda: fake_llm)
    _, idx = _index()
    result = ask(idx, "why is DNS resolution timing out?")

    assert calls["n"] == 2
    assert result["_iterations"] == 2
    assert result["evidence"]
    assert result["owner"] != "unknown"
