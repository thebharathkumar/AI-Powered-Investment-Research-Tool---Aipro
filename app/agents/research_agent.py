from __future__ import annotations

from typing import Any, TypedDict, Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import Tool
from langgraph.graph import StateGraph, END

from app.rag.pipeline import query_rag
from app.config import get_settings

settings = get_settings()

RESEARCH_SYSTEM_PROMPT = """You are an expert investment research analyst. Your role is to:
1. Analyze financial documents, SEC filings, and regulatory data
2. Extract key insights about company performance, risks, and opportunities
3. Provide structured, evidence-based analysis with confidence levels
4. Identify potential red flags or concerns in financial statements
5. Compare metrics against industry benchmarks when relevant

Always be factual, cite sources, and clearly distinguish between facts and inferences."""


class ResearchState(TypedDict):
    query: str
    ticker: str
    messages: Annotated[list, operator.add]
    retrieved_context: str
    analysis: str
    sources: list[dict]
    confidence_score: float
    step_count: int


def retrieve_financial_context(query: str) -> str:
    result = query_rag(query)
    return result["answer"]


def build_research_graph() -> StateGraph:
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.0,
        openai_api_key=settings.openai_api_key,
    )

    tools = [
        Tool(
            name="retrieve_financial_context",
            func=retrieve_financial_context,
            description="Retrieve relevant financial context from the document database using semantic search.",
        )
    ]

    def retrieve_node(state: ResearchState) -> dict:
        query = f"{state['ticker']} {state['query']}" if state.get("ticker") else state["query"]
        result = query_rag(query)
        return {
            "retrieved_context": result["answer"],
            "sources": result["sources"],
            "messages": [HumanMessage(content=f"Retrieved context: {result['answer'][:500]}...")],
            "step_count": state.get("step_count", 0) + 1,
        }

    def analyze_node(state: ResearchState) -> dict:
        context = state.get("retrieved_context", "")
        query = state["query"]
        ticker = state.get("ticker", "")

        messages = [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""Ticker: {ticker}
Query: {query}

Retrieved Context:
{context}

Provide a comprehensive investment research analysis based on this context.
Include: key findings, risk factors, financial highlights, and overall assessment.
End with a confidence score (0.0-1.0) for your analysis based on available evidence."""
            ),
        ]

        response = llm.invoke(messages)
        analysis_text = response.content

        confidence = 0.7
        if "confidence score:" in analysis_text.lower():
            try:
                parts = analysis_text.lower().split("confidence score:")
                score_str = parts[-1].strip().split()[0].rstrip(".")
                confidence = float(score_str)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, IndexError):
                pass

        return {
            "analysis": analysis_text,
            "confidence_score": confidence,
            "messages": [response],
            "step_count": state.get("step_count", 0) + 1,
        }

    def should_continue(state: ResearchState) -> str:
        if state.get("step_count", 0) >= 3:
            return "end"
        if state.get("retrieved_context") and not state.get("analysis"):
            return "analyze"
        if not state.get("retrieved_context"):
            return "retrieve"
        return "end"

    graph = StateGraph(ResearchState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        should_continue,
        {"analyze": "analyze", "retrieve": "retrieve", "end": END},
    )
    graph.add_edge("analyze", END)

    return graph.compile()


def run_research(query: str, ticker: str = "") -> dict[str, Any]:
    graph = build_research_graph()
    initial_state: ResearchState = {
        "query": query,
        "ticker": ticker,
        "messages": [],
        "retrieved_context": "",
        "analysis": "",
        "sources": [],
        "confidence_score": 0.0,
        "step_count": 0,
    }
    final_state = graph.invoke(initial_state)
    return {
        "query": query,
        "ticker": ticker,
        "analysis": final_state.get("analysis", ""),
        "sources": final_state.get("sources", []),
        "confidence_score": final_state.get("confidence_score", 0.0),
    }
