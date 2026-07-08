"""CI triage agent graph — orchestrates four specialized agents.

Graph:  retrieve -> diagnose -> critique -> (refine -> retrieve)* -> enrich -> done

Retriever, Diagnosis, Critic, and Tool agents (see agents.py) each own
one responsibility. The Diagnosis Agent proposes a grounded answer;
the Critic Agent independently verifies it — citations actually
retrieved, root-cause claims actually supported by the cited text,
and confidence — then either accepts or sends the Retriever Agent a
refined query. That accept/refine exchange is the collaboration: a
bounded loop, not an open one — MAX_ITERATIONS guards against the
classic agent failure mode of looping forever. Once accepted, the
Tool Agent enriches the answer via MCP tool calls.

LLM backend: ChatAnthropic when ANTHROPIC_API_KEY is set; otherwise a
deterministic MockLLM so the graph, retrieval, and evals run offline.
"""
from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from .agents import CriticAgent, DiagnosisAgent, RetrieverAgent, ToolAgent
from .retrieve import HybridIndex

MAX_ITERATIONS = 3


class TriageState(TypedDict, total=False):
    question: str
    query: str
    chunks: list
    answer: dict
    critic_feedback: str
    accepted: bool
    refined_query: str
    iterations: int


def build_agent(index: HybridIndex):
    retriever = RetrieverAgent(index)
    diagnosis = DiagnosisAgent()
    critic = CriticAgent()
    tool_agent = ToolAgent()

    def retrieve(state: TriageState) -> TriageState:
        query = state.get("query") or state["question"]
        chunks = retriever.run(query)
        return {"chunks": chunks, "iterations": state.get("iterations", 0) + 1}

    def diagnose(state: TriageState) -> TriageState:
        return {"answer": diagnosis.run(state["question"], state["chunks"])}

    def critique(state: TriageState) -> TriageState:
        return critic.run(state["answer"], state["chunks"])

    def route(state: TriageState) -> Literal["refine", "enrich"]:
        if state.get("accepted") or state["iterations"] >= MAX_ITERATIONS:
            return "enrich"
        return "refine"

    def refine(state: TriageState) -> TriageState:
        return {"query": state["refined_query"]}

    def enrich(state: TriageState) -> TriageState:
        return {"answer": tool_agent.run(state["answer"], state["chunks"])}

    g = StateGraph(TriageState)
    g.add_node("retrieve", retrieve)
    g.add_node("diagnose", diagnose)
    g.add_node("critique", critique)
    g.add_node("refine", refine)
    g.add_node("enrich", enrich)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "diagnose")
    g.add_edge("diagnose", "critique")
    g.add_conditional_edges("critique", route, {"refine": "refine", "enrich": "enrich"})
    g.add_edge("refine", "retrieve")
    g.add_edge("enrich", END)
    return g.compile()


def ask(index: HybridIndex, question: str) -> dict:
    agent = build_agent(index)
    final = agent.invoke({"question": question})
    result = final["answer"]
    result["_retrieved"] = [c.chunk_id for c in final["chunks"]]
    result["_iterations"] = final["iterations"]
    result["_critic_feedback"] = final.get("critic_feedback", "")
    return result
