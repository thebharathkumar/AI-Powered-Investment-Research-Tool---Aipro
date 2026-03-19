from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.rag.pipeline import ingest_document
from app.config import get_settings

settings = get_settings()

EXTRACTION_PROMPT = """You are a financial document analyst. Extract structured information from the provided financial document excerpt.

Extract and return a JSON object with the following fields (use null if not found):
- document_type: Type of document (10-K, 10-Q, 8-K, Annual Report, etc.)
- company_name: Company name
- ticker: Stock ticker symbol
- period: Reporting period (e.g., "FY2023", "Q3 2023")
- key_metrics: Object with financial metrics (revenue, net_income, eps, etc.)
- risk_factors: List of key risk factors mentioned
- management_highlights: List of key management discussion points
- forward_guidance: Any forward-looking statements
- sentiment: Overall document sentiment (positive/neutral/negative)"""


class DocumentAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.0,
            openai_api_key=settings.openai_api_key,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
        )

    def extract_metadata(self, content: str) -> dict[str, Any]:
        excerpt = content[:8000]
        messages = [
            SystemMessage(content=EXTRACTION_PROMPT),
            HumanMessage(content=f"Document excerpt:\n\n{excerpt}\n\nReturn only valid JSON."),
        ]
        response = self.llm.invoke(messages)
        import json
        try:
            text = response.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except (json.JSONDecodeError, IndexError):
            return {"raw_response": response.content}

    def process_document(
        self,
        content: str,
        ticker: str,
        doc_type: str,
        filing_date: str,
        source_url: str = "",
    ) -> dict[str, Any]:
        metadata = {
            "ticker": ticker,
            "doc_type": doc_type,
            "filing_date": filing_date,
            "source_url": source_url,
        }
        extracted = self.extract_metadata(content)
        metadata.update({k: v for k, v in extracted.items() if v is not None})

        ids = ingest_document(content, metadata)

        return {
            "status": "processed",
            "chunk_count": len(ids),
            "embedding_ids": ids,
            "extracted_metadata": extracted,
        }

    def summarize_document(self, content: str, max_length: int = 500) -> str:
        messages = [
            SystemMessage(
                content="You are a financial analyst. Summarize the key points of this financial document concisely."
            ),
            HumanMessage(
                content=f"Document (first 6000 chars):\n\n{content[:6000]}\n\nProvide a {max_length}-word summary."
            ),
        ]
        response = self.llm.invoke(messages)
        return response.content
