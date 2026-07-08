"""Specialized agents that collaborate over the LangGraph state machine
in agent.py.

RetrieverAgent  — hybrid search over the log index.
DiagnosisAgent  — LLM: proposes a grounded root-cause diagnosis.
CriticAgent     — rule-based verifier: independently checks the
                  diagnosis (citations actually retrieved + root-cause
                  claims actually supported by the cited text +
                  confidence) and decides accept/refine. Deliberately
                  not another LLM call — you can't reliably ask a model
                  to grade its own hallucinations, so grounding is
                  checked programmatically.
ToolAgent       — enriches an accepted answer via MCP tool calls
                  (test owner, build artifact metadata).

Diagnosis proposes, Critic disposes: the accept/refine loop between
them (bounded by MAX_ITERATIONS in agent.py) is the collaboration.
"""
from __future__ import annotations

import json
import os
import re

from .chunk import Chunk
from .mcp import client as mcp_client
from .retrieve import HybridIndex

DIAGNOSIS_PROMPT = """You are the Diagnosis Agent on a CI failure triage team, for tmt / Fedora CI logs.

Question: {question}

Retrieved log chunks (each begins with its chunk_id):
{context}

Respond with JSON only:
{{
  "failing_test": "<test name or 'unknown'>",
  "root_cause": "<one-paragraph diagnosis grounded in the chunks>",
  "evidence": ["<chunk_id>", ...],
  "confidence": <0.0-1.0>,
  "refined_query": "<better search query if unsure, else ''>"
}}
Only cite chunk_ids that appear above. If the chunks are insufficient,
say so and lower confidence rather than guessing. A Critic agent will
independently verify every citation and claim you make."""

_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]{3,}")


class MockLLM:
    """Offline stand-in for the Diagnosis Agent: picks the top failing
    chunk deterministically and quotes its own text, so the Critic
    Agent's lexical-grounding check passes without a real LLM. Lets
    the graph and eval harness run without an API key."""

    def invoke(self, prompt: str) -> str:
        ids = re.findall(r"\[chunk_id=([^\]]+)\]", prompt)
        top = ids[0] if ids else "unknown"
        # if the question names a specific run, prefer a candidate from
        # that run over whatever ranked first — a real LLM would read
        # the question; the mock needs this nudge to do the same.
        q_match = re.search(r"Question: (.*)", prompt)
        run_match = re.search(r"run-\d+", q_match.group(1)) if q_match else None
        if run_match:
            named = next((i for i in ids if i.startswith(run_match.group(0) + "|")), None)
            top = named or top
        test = top.split("|")[1] if "|" in top else "unknown"
        m = re.search(
            r"\[chunk_id=" + re.escape(top) + r"\][^\n]*\n(.*?)(?=\n\[chunk_id=|\Z)",
            prompt,
            re.S,
        )
        snippet = " ".join((m.group(1) if m else "").split())[:300]
        return json.dumps(
            {
                "failing_test": test,
                "root_cause": f"Mock diagnosis grounded in {top}: {snippet} "
                "(set ANTHROPIC_API_KEY for real diagnosis).",
                "evidence": [top],
                "confidence": 0.65,
                "refined_query": "",
            }
        )


def _get_llm():
    if os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        chat = ChatAnthropic(model="claude-sonnet-4-5", max_tokens=1024)
        return lambda prompt: chat.invoke(prompt).content
    return MockLLM().invoke


class RetrieverAgent:
    """Owns hybrid search over the log index."""

    def __init__(self, index: HybridIndex) -> None:
        self.index = index

    def run(self, query: str) -> list[Chunk]:
        return self.index.search(query, k=6, failed_only=True)


class DiagnosisAgent:
    """Owns proposing a grounded root-cause diagnosis from retrieved chunks."""

    def __init__(self) -> None:
        self._llm = _get_llm()

    def run(self, question: str, chunks: list[Chunk]) -> dict:
        context = "\n\n".join(
            f"[chunk_id={c.chunk_id}] (test={c.test_name}, failed={c.failed})\n{c.text[:1200]}"
            for c in chunks
        )
        raw = self._llm(DIAGNOSIS_PROMPT.format(question=question, context=context))
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "failing_test": "unknown",
                "root_cause": raw[:500],
                "evidence": [],
                "confidence": 0.0,
                "refined_query": "",
            }


class CriticAgent:
    """Owns independent verification of a Diagnosis Agent draft."""

    MIN_CONFIDENCE = 0.6
    MIN_OVERLAP = 0.15

    def run(self, answer: dict, chunks: list[Chunk]) -> dict:
        shown = {c.chunk_id: c for c in chunks}
        cited_ids = [e for e in answer.get("evidence", []) if e in shown]
        dropped = set(answer.get("evidence", [])) - set(cited_ids)
        answer = {**answer, "evidence": cited_ids}

        evidence_text = " ".join(shown[cid].text for cid in cited_ids)
        overlap = self._overlap(answer.get("root_cause", ""), evidence_text)

        reasons = []
        if dropped:
            reasons.append(f"dropped unverifiable citations: {sorted(dropped)}")
        if not cited_ids:
            reasons.append("no valid citations")
        if cited_ids and overlap < self.MIN_OVERLAP:
            reasons.append(f"root_cause not well grounded in cited text (overlap={overlap:.2f})")
        if answer.get("confidence", 0) < self.MIN_CONFIDENCE:
            reasons.append(f"low confidence ({answer.get('confidence', 0):.2f})")

        accepted = not reasons
        refined_query = answer.get("refined_query") or ""
        if not accepted and not refined_query:
            missing = self._salient_terms(answer.get("root_cause", "")) - self._salient_terms(evidence_text)
            refined_query = " ".join(sorted(missing)) or answer.get("failing_test", "")

        return {
            "answer": answer,
            "critic_feedback": "; ".join(reasons) if reasons else "grounded and confident",
            "accepted": accepted,
            "refined_query": refined_query,
        }

    def _salient_terms(self, text: str) -> set[str]:
        return {w.lower() for w in _WORD_RE.findall(text)}

    def _overlap(self, claim: str, evidence_text: str) -> float:
        claim_terms = self._salient_terms(claim)
        if not claim_terms:
            return 1.0
        evidence_terms = self._salient_terms(evidence_text)
        return len(claim_terms & evidence_terms) / len(claim_terms)


class ToolAgent:
    """Owns enriching an accepted answer via MCP tool calls. Any tool
    registered on the MCP server (src/ci_triage/mcp/server.py) is
    reachable here with no server-side code change required."""

    def run(self, answer: dict, chunks: list[Chunk]) -> dict:
        evidence = answer.get("evidence") or []
        run_id = evidence[0].split("|")[0] if evidence else (chunks[0].run_id if chunks else None)

        calls: list[tuple[str, dict]] = [
            ("lookup_test_owner", {"test_name": answer.get("failing_test", "")})
        ]
        if run_id:
            calls.append(("fetch_build_artifact", {"run_id": run_id}))

        try:
            results = mcp_client.call_many(calls)
        except Exception as exc:  # MCP server unavailable — degrade gracefully
            return {**answer, "tool_error": str(exc)}

        answer = {**answer, "owner": results[0].get("owner", "unknown")}
        if run_id and len(results) > 1:
            answer["artifact_info"] = results[1]
        return answer
