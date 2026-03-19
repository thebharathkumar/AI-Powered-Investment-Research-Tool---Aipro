import pytest
from unittest.mock import MagicMock, patch


class TestResearchAgent:
    @patch("app.agents.research_agent.query_rag")
    def test_run_research_returns_expected_keys(self, mock_rag):
        mock_rag.return_value = {
            "answer": "Revenue grew 10% YoY to $100B driven by cloud services.",
            "sources": [{"content": "Revenue was $100B", "metadata": {"ticker": "MSFT"}}],
        }
        from app.agents.research_agent import run_research

        with patch("app.agents.research_agent.ChatOpenAI") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Analysis: Strong revenue growth. Confidence score: 0.85"
            mock_llm.invoke.return_value = mock_response
            mock_llm_cls.return_value = mock_llm

            result = run_research("What is the revenue growth?", ticker="MSFT")

        assert "query" in result
        assert "analysis" in result
        assert "sources" in result
        assert "confidence_score" in result
        assert result["ticker"] == "MSFT"


class TestDocumentAgent:
    @patch("app.agents.document_agent.ChatOpenAI")
    def test_extract_metadata_returns_dict(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '{"document_type": "10-K", "company_name": "Apple Inc.", '
            '"ticker": "AAPL", "period": "FY2023", "sentiment": "positive"}'
        )
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        from app.agents.document_agent import DocumentAgent

        agent = DocumentAgent()
        result = agent.extract_metadata("Apple Inc. annual revenue was $383 billion in FY2023.")
        assert isinstance(result, dict)

    @patch("app.agents.document_agent.ChatOpenAI")
    @patch("app.agents.document_agent.ingest_document")
    def test_process_document(self, mock_ingest, mock_llm_cls):
        mock_ingest.return_value = ["id1", "id2", "id3"]
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"document_type": "10-K", "sentiment": "positive"}'
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        from app.agents.document_agent import DocumentAgent

        agent = DocumentAgent()
        result = agent.process_document(
            content="Apple annual report 2023. Revenue $383B.",
            ticker="AAPL",
            doc_type="10-K",
            filing_date="2023-11-02",
        )
        assert result["status"] == "processed"
        assert result["chunk_count"] == 3


class TestAnalysisAgent:
    @patch("app.agents.analysis_agent.ChatOpenAI")
    def test_analyze_sentiment(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '{"sentiment": "positive", "score": 0.7, '
            '"key_drivers": ["revenue growth"], "uncertainty_level": "low"}'
        )
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        from app.agents.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()
        result = agent.analyze_sentiment("Revenue grew 15% year-over-year.")
        assert isinstance(result, dict)
        assert "sentiment" in result

    @patch("app.agents.analysis_agent.ChatOpenAI")
    def test_assess_risks(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '{"overall_risk_level": "medium", "top_risks": ["regulatory risk", "market competition"]}'
        )
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        from app.agents.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()
        result = agent.assess_risks("The company faces significant regulatory headwinds.")
        assert isinstance(result, dict)

    @patch("app.agents.analysis_agent.ChatOpenAI")
    def test_compare_quarters(self, mock_llm_cls):
        from app.agents.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()
        current = {"revenue": 110.0, "net_income": 11.0}
        prior = {"revenue": 100.0, "net_income": 10.0}
        result = agent.compare_quarters("AAPL", current, prior)
        assert result["ticker"] == "AAPL"
        assert result["metric_changes"]["revenue"]["pct_change"] == pytest.approx(10.0)
