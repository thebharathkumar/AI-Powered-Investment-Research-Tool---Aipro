from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage

from app.config import get_settings

settings = get_settings()

SENTIMENT_PROMPT = """Analyze the financial text and return a JSON object with:
- sentiment: "positive", "neutral", or "negative"
- score: float from -1.0 (very negative) to 1.0 (very positive)
- key_drivers: list of 3-5 main sentiment drivers
- uncertainty_level: "low", "medium", or "high"
Return only valid JSON."""

RISK_ASSESSMENT_PROMPT = """Analyze the financial text and identify risks. Return a JSON object with:
- overall_risk_level: "low", "medium", "high", or "critical"
- risk_categories: object with keys (market_risk, credit_risk, operational_risk, regulatory_risk, liquidity_risk) each with "level" and "description"
- top_risks: list of top 5 specific risk factors
- mitigating_factors: list of risk mitigation factors mentioned
Return only valid JSON."""

FINANCIAL_SUMMARY_PROMPT = """You are a senior financial analyst at a top investment bank.
Analyze the provided financial data and generate a comprehensive investment research report.

Include:
1. Executive Summary (3-4 sentences)
2. Financial Performance Analysis (key metrics and trends)
3. Competitive Position Assessment
4. Risk Factors (top 5 with severity)
5. Growth Catalysts
6. Valuation Perspective
7. Investment Recommendation (Buy/Hold/Sell with rationale)

Be specific, data-driven, and avoid speculation without evidence."""


class AnalysisAgent:
    def __init__(self, use_claude: bool = False):
        if use_claude and settings.anthropic_api_key:
            self.llm = ChatAnthropic(
                model=settings.anthropic_model,
                anthropic_api_key=settings.anthropic_api_key,
                temperature=0.0,
            )
        else:
            self.llm = ChatOpenAI(
                model=settings.openai_model,
                temperature=0.0,
                openai_api_key=settings.openai_api_key,
            )

    def _parse_json_response(self, content: str) -> dict:
        import json
        try:
            text = content.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except (json.JSONDecodeError, IndexError):
            return {"raw_response": content}

    def analyze_sentiment(self, text: str) -> dict[str, Any]:
        messages = [
            SystemMessage(content=SENTIMENT_PROMPT),
            HumanMessage(content=f"Financial text:\n\n{text[:4000]}"),
        ]
        response = self.llm.invoke(messages)
        return self._parse_json_response(response.content)

    def assess_risks(self, text: str) -> dict[str, Any]:
        messages = [
            SystemMessage(content=RISK_ASSESSMENT_PROMPT),
            HumanMessage(content=f"Financial text:\n\n{text[:4000]}"),
        ]
        response = self.llm.invoke(messages)
        return self._parse_json_response(response.content)

    def generate_research_report(
        self,
        ticker: str,
        company_name: str,
        financial_data: dict[str, Any],
        context_text: str = "",
    ) -> str:
        data_str = "\n".join([f"- {k}: {v}" for k, v in financial_data.items()])
        messages = [
            SystemMessage(content=FINANCIAL_SUMMARY_PROMPT),
            HumanMessage(
                content=f"""Company: {company_name} ({ticker})

Financial Data:
{data_str}

Additional Context:
{context_text[:3000] if context_text else "No additional context provided."}

Generate a comprehensive investment research report."""
            ),
        ]
        response = self.llm.invoke(messages)
        return response.content

    def compare_quarters(
        self,
        ticker: str,
        current_metrics: dict[str, float],
        prior_metrics: dict[str, float],
    ) -> dict[str, Any]:
        changes = {}
        for key in current_metrics:
            if key in prior_metrics and prior_metrics[key] and prior_metrics[key] != 0:
                pct_change = ((current_metrics[key] - prior_metrics[key]) / abs(prior_metrics[key])) * 100
                changes[key] = {
                    "current": current_metrics[key],
                    "prior": prior_metrics[key],
                    "pct_change": round(pct_change, 2),
                    "direction": "up" if pct_change > 0 else "down",
                }
        return {"ticker": ticker, "metric_changes": changes}
