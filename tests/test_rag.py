import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document as LCDocument

from app.rag.pipeline import chunk_document, get_text_splitter, FINANCIAL_RAG_PROMPT


class TestRAGPipeline:
    def test_chunk_document_basic(self):
        content = "This is a test financial document. " * 100
        metadata = {"ticker": "AAPL", "doc_type": "10-K"}
        chunks = chunk_document(content, metadata)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, LCDocument)
            assert chunk.metadata["ticker"] == "AAPL"
            assert "chunk_index" in chunk.metadata
            assert "content_hash" in chunk.metadata

    def test_chunk_document_preserves_metadata(self):
        content = "Revenue grew 15% year-over-year. Net income increased significantly." * 50
        metadata = {"ticker": "MSFT", "doc_type": "10-Q", "filing_date": "2023-10-01"}
        chunks = chunk_document(content, metadata)
        for chunk in chunks:
            assert chunk.metadata["doc_type"] == "10-Q"
            assert chunk.metadata["filing_date"] == "2023-10-01"

    def test_chunk_indices_sequential(self):
        content = "A" * 5000
        metadata = {"ticker": "GOOG"}
        chunks = chunk_document(content, metadata)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_text_splitter_config(self):
        splitter = get_text_splitter()
        assert splitter._chunk_size == 1000
        assert splitter._chunk_overlap == 200

    def test_financial_rag_prompt_has_required_vars(self):
        assert "context" in FINANCIAL_RAG_PROMPT.input_variables
        assert "input" in FINANCIAL_RAG_PROMPT.input_variables

    def test_financial_rag_prompt_no_fabrication_instruction(self):
        full_text = " ".join(
            msg.prompt.template.lower() if hasattr(msg, "prompt") else str(msg).lower()
            for msg in FINANCIAL_RAG_PROMPT.messages
        )
        assert "fabricate" in full_text or "never" in full_text
