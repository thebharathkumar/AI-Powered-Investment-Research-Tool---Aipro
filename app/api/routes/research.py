from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models import ResearchQuery
from app.db.queries import get_recent_queries, save_research_query, get_ticker_analysis_summary
from app.agents.research_agent import run_research
from app.rag.pipeline import query_rag

router = APIRouter(prefix="/research", tags=["research"])


class ResearchRequest(BaseModel):
    query: str
    ticker: Optional[str] = None
    use_agent: bool = True


class ResearchResponse(BaseModel):
    query: str
    ticker: Optional[str] = None
    analysis: str
    sources: list[dict]
    confidence_score: float


@router.post("/query", response_model=ResearchResponse)
async def research_query(
    request: ResearchRequest,
    db: AsyncSession = Depends(get_db),
):
    import time

    start_ms = int(time.time() * 1000)

    if request.use_agent:
        result = run_research(query=request.query, ticker=request.ticker or "")
    else:
        raw = query_rag(request.query)
        result = {
            "query": request.query,
            "ticker": request.ticker,
            "analysis": raw["answer"],
            "sources": raw["sources"],
            "confidence_score": 0.7,
        }

    latency_ms = int(time.time() * 1000) - start_ms

    rq = ResearchQuery(
        ticker=request.ticker,
        query_text=request.query,
        query_type="agent" if request.use_agent else "rag",
        response_text=result["analysis"],
        sources_used=result["sources"],
        confidence_score=result["confidence_score"],
        latency_ms=latency_ms,
    )
    await save_research_query(db, rq)

    return ResearchResponse(**result)


@router.get("/history", response_model=list[dict])
async def query_history(
    ticker: Optional[str] = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    queries = await get_recent_queries(db, ticker=ticker, limit=limit)
    return [
        {
            "id": q.id,
            "ticker": q.ticker,
            "query_text": q.query_text,
            "query_type": q.query_type,
            "confidence_score": q.confidence_score,
            "created_at": q.created_at.isoformat(),
        }
        for q in queries
    ]


@router.get("/summary/{ticker}")
async def ticker_summary(ticker: str, db: AsyncSession = Depends(get_db)):
    return await get_ticker_analysis_summary(db, ticker)
