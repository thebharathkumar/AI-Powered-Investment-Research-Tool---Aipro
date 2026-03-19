import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestAnalysisEndpoints:
    @patch("app.api.routes.analysis.AnalysisAgent")
    def test_sentiment_endpoint(self, mock_agent_cls, client):
        mock_agent = MagicMock()
        mock_agent.analyze_sentiment.return_value = {
            "sentiment": "positive",
            "score": 0.75,
            "key_drivers": ["revenue growth"],
            "uncertainty_level": "low",
        }
        mock_agent_cls.return_value = mock_agent

        resp = client.post(
            "/api/v1/analysis/sentiment",
            json={"text": "Revenue grew significantly this quarter.", "use_claude": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "sentiment" in data

    @patch("app.api.routes.analysis.AnalysisAgent")
    def test_risks_endpoint(self, mock_agent_cls, client):
        mock_agent = MagicMock()
        mock_agent.assess_risks.return_value = {
            "overall_risk_level": "medium",
            "top_risks": ["competition", "regulation"],
        }
        mock_agent_cls.return_value = mock_agent

        resp = client.post(
            "/api/v1/analysis/risks",
            json={"text": "Regulatory headwinds and competition pose risks.", "use_claude": False},
        )
        assert resp.status_code == 200

    def test_statistics_endpoint(self, client):
        prices = [100.0, 105.0, 102.0, 108.0, 110.0, 107.0, 115.0]
        resp = client.post(
            "/api/v1/analysis/statistics",
            json={"prices": prices, "risk_free_rate": 0.05},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "volatility_annualized" in data
        assert "sharpe_ratio" in data
        assert "max_drawdown" in data
        assert "linear_trend" in data

    def test_statistics_too_few_prices(self, client):
        resp = client.post(
            "/api/v1/analysis/statistics",
            json={"prices": [100.0], "risk_free_rate": 0.05},
        )
        assert resp.status_code == 400

    def test_hallucination_check_endpoint(self, client):
        resp = client.post(
            "/api/v1/analysis/hallucination-check",
            json={
                "context_chunks": ["Revenue was $100 million in Q3 2023"],
                "response": "Based on the filing, revenue was $100 million.",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "reliability_score" in data
        assert "assessment" in data
