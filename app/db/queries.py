from datetime import datetime
from typing import Optional
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Document, DocumentChunk, ResearchQuery, ResearchAnalysis, FinancialMetric


async def get_documents_by_ticker(
    db: AsyncSession,
    ticker: str,
    doc_type: Optional[str] = None,
    limit: int = 20,
) -> list[Document]:
    stmt = select(Document).where(Document.ticker == ticker.upper())
    if doc_type:
        stmt = stmt.where(Document.doc_type == doc_type)
    stmt = stmt.order_by(Document.filing_date.desc()).limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_document_chunks(db: AsyncSession, document_id: int) -> list[DocumentChunk]:
    stmt = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_recent_queries(
    db: AsyncSession,
    ticker: Optional[str] = None,
    limit: int = 50,
) -> list[ResearchQuery]:
    stmt = select(ResearchQuery)
    if ticker:
        stmt = stmt.where(ResearchQuery.ticker == ticker.upper())
    stmt = stmt.order_by(ResearchQuery.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_financial_metrics_timeseries(
    db: AsyncSession,
    ticker: str,
    start_date: datetime,
    end_date: datetime,
) -> list[FinancialMetric]:
    stmt = (
        select(FinancialMetric)
        .where(
            and_(
                FinancialMetric.ticker == ticker.upper(),
                FinancialMetric.metric_date >= start_date,
                FinancialMetric.metric_date <= end_date,
            )
        )
        .order_by(FinancialMetric.metric_date.asc())
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_ticker_analysis_summary(db: AsyncSession, ticker: str) -> dict:
    """Aggregated analysis summary for a ticker."""
    doc_count_stmt = select(func.count(Document.id)).where(Document.ticker == ticker.upper())
    doc_count = (await db.execute(doc_count_stmt)).scalar_one()

    analysis_count_stmt = (
        select(func.count(ResearchAnalysis.id))
        .join(Document, ResearchAnalysis.document_id == Document.id)
        .where(Document.ticker == ticker.upper())
    )
    analysis_count = (await db.execute(analysis_count_stmt)).scalar_one()

    avg_confidence_stmt = (
        select(func.avg(ResearchAnalysis.confidence_score))
        .join(Document, ResearchAnalysis.document_id == Document.id)
        .where(Document.ticker == ticker.upper())
    )
    avg_confidence = (await db.execute(avg_confidence_stmt)).scalar_one()

    return {
        "ticker": ticker.upper(),
        "document_count": doc_count,
        "analysis_count": analysis_count,
        "avg_confidence_score": float(avg_confidence) if avg_confidence else None,
    }


async def upsert_financial_metric(db: AsyncSession, metric: FinancialMetric) -> FinancialMetric:
    db.add(metric)
    await db.flush()
    return metric


async def save_research_query(db: AsyncSession, query: ResearchQuery) -> ResearchQuery:
    db.add(query)
    await db.flush()
    return query
