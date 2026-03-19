from __future__ import annotations

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models import Document, DocumentChunk
from app.db.queries import get_documents_by_ticker, get_document_chunks
from app.agents.document_agent import DocumentAgent

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    id: int
    ticker: str
    company_name: str
    doc_type: str
    filing_date: datetime
    source_url: Optional[str] = None
    page_count: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


class IngestRequest(BaseModel):
    ticker: str
    company_name: str
    doc_type: str
    filing_date: str
    content: str
    source_url: Optional[str] = None


class IngestResponse(BaseModel):
    status: str
    document_id: Optional[int] = None
    chunk_count: int
    extracted_metadata: dict


@router.get("/{ticker}", response_model=list[DocumentResponse])
async def list_documents(
    ticker: str,
    doc_type: Optional[str] = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    documents = await get_documents_by_ticker(db, ticker, doc_type=doc_type, limit=limit)
    return documents


@router.get("/{ticker}/chunks/{document_id}", response_model=list[dict])
async def get_chunks(
    ticker: str,
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    chunks = await get_document_chunks(db, document_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="Document chunks not found")
    return [
        {
            "id": c.id,
            "chunk_index": c.chunk_index,
            "content": c.content[:500],
            "token_count": c.token_count,
        }
        for c in chunks
    ]


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document_endpoint(
    request: IngestRequest,
    db: AsyncSession = Depends(get_db),
):
    agent = DocumentAgent()
    result = agent.process_document(
        content=request.content,
        ticker=request.ticker,
        doc_type=request.doc_type,
        filing_date=request.filing_date,
        source_url=request.source_url or "",
    )

    doc = Document(
        ticker=request.ticker.upper(),
        company_name=request.company_name,
        doc_type=request.doc_type,
        filing_date=datetime.fromisoformat(request.filing_date),
        source_url=request.source_url,
        page_count=result.get("chunk_count"),
    )
    db.add(doc)
    await db.flush()

    for i, eid in enumerate(result["embedding_ids"]):
        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=i,
            content="",
            embedding_id=eid,
        )
        db.add(chunk)

    await db.flush()

    return IngestResponse(
        status="success",
        document_id=doc.id,
        chunk_count=result["chunk_count"],
        extracted_metadata=result["extracted_metadata"],
    )
