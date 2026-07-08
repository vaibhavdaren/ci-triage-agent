"""CI triage agent built on LangGraph.

Graph:  retrieve -> triage -> (refine -> retrieve)* -> done

The triage node answers from retrieved chunks and must ground every
claim in a chunk_id citation. If it reports low confidence, the
refine node rewrites the query (e.g. adds the suspect test name) and
we retrieve again — a bounded self-correction loop, not an open one:
MAX_ITERATIONS guards against the classic agent failure mode of
looping forever.

LLM backend: ChatAnthropic when ANTHROPIC_API_KEY is set; otherwise a
deterministic MockLLM so the graph, retrieval, and evals run offline.
"""
from __future__ import annotations

import json
import os
import re
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from .retrieve import HybridIndex

MAX_ITERATIONS = 3

TRIAGE_PROMPT = """You are a CI failure triage assistant for tmt / Fedora CI logs.

Question: {question}

Retrieved log chunks (each begins with its chunk_id):
{context}

Respond with JSON only:
{{
  "failing_test": "<test name or 'unknown'>",
  "root_cause": "<one-paragraph diagnosis grounded in the chunks>",
  "evidence": ["<chunk_id>", ...],
  "confidence": <0.0-1.0>,
  "refined_query": "<better search query if confidence < 0.6, else ''>"
}}
Only cite chunk_ids that appear above. If the chunks are insufficient,
say so and lower confidence rather than guessing."""


class TriageState(TypedDict, total=False):
    question: str
    query: str
    chunks: list
    answer: dict
    iterations: int


class MockLLM:
    """Offline stand-in: picks the top failing chunk deterministically.
    Lets the graph and eval harness run without an API key."""

    def invoke(self, prompt: str) -> str:
        ids = re.findall(r"\[chunk_id=([^\]]+)\]", prompt)
        top = ids[0] if ids else "unknown"
        test = top.split("|")[1] if "|" in top else "unknown"
        return json.dumps(
            {
                "failing_test": test,
                "root_cause": "Top-ranked failing chunk selected by mock backend "
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


def build_agent(index: HybridIndex):
    llm = _get_llm()

    def retrieve(state: TriageState) -> TriageState:
        query = state.get("query") or state["question"]
        chunks = index.search(query, k=6, failed_only=True)
        return {"chunks": chunks, "iterations": state.get("iterations", 0) + 1}

    def triage(state: TriageState) -> TriageState:
        context = "\n\n".join(
            f"[chunk_id={c.chunk_id}] (test={c.test_name}, failed={c.failed})\n{c.text[:1200]}"
            for c in state["chunks"]
        )
        raw = llm(TRIAGE_PROMPT.format(question=state["question"], context=context))
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        try:
            answer = json.loads(raw)
        except json.JSONDecodeError:
            answer = {"failing_test": "unknown", "root_cause": raw[:500],
                      "evidence": [], "confidence": 0.0, "refined_query": ""}
        # Groundedness guard: drop citations of chunks we never showed.
        shown = {c.chunk_id for c in state["chunks"]}
        answer["evidence"] = [e for e in answer.get("evidence", []) if e in shown]
        return {"answer": answer}

    def should_refine(state: TriageState) -> Literal["refine", "done"]:
        a = state["answer"]
        if (
            a.get("confidence", 0) < 0.6
            and a.get("refined_query")
            and state["iterations"] < MAX_ITERATIONS
        ):
            return "refine"
        return "done"

    def refine(state: TriageState) -> TriageState:
        return {"query": state["answer"]["refined_query"]}

    g = StateGraph(TriageState)
    g.add_node("retrieve", retrieve)
    g.add_node("triage", triage)
    g.add_node("refine", refine)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "triage")
    g.add_conditional_edges("triage", should_refine, {"refine": "refine", "done": END})
    g.add_edge("refine", "retrieve")
    return g.compile()


def ask(index: HybridIndex, question: str) -> dict:
    agent = build_agent(index)
    final = agent.invoke({"question": question})
    result = final["answer"]
    result["_retrieved"] = [c.chunk_id for c in final["chunks"]]
    result["_iterations"] = final["iterations"]
    return result
