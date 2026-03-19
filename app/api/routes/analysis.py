from __future__ import annotations

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models import FinancialMetric
from app.db.queries import get_financial_metrics_timeseries, upsert_financial_metric
from app.agents.analysis_agent import AnalysisAgent
from app.analytics.statistical_analysis import (
    descriptive_stats,
    compute_volatility,
    compute_sharpe_ratio,
    compute_max_drawdown,
    linear_trend,
)
from app.analytics.feature_engineering import engineer_hallucination_reduction_features

router = APIRouter(prefix="/analysis", tags=["analysis"])


class SentimentRequest(BaseModel):
    text: str
    use_claude: bool = False


class RiskRequest(BaseModel):
    text: str
    use_claude: bool = False


class ReportRequest(BaseModel):
    ticker: str
    company_name: str
    financial_data: dict
    context_text: Optional[str] = None
    use_claude: bool = False


class MetricsRequest(BaseModel):
    ticker: str
    metric_date: str
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    eps: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    operating_margin: Optional[float] = None
    free_cash_flow: Optional[float] = None
    extra_metrics: Optional[dict] = None


class StatisticsRequest(BaseModel):
    prices: list[float]
    risk_free_rate: float = 0.05


@router.post("/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    agent = AnalysisAgent(use_claude=request.use_claude)
    return agent.analyze_sentiment(request.text)


@router.post("/risks")
async def assess_risks(request: RiskRequest):
    agent = AnalysisAgent(use_claude=request.use_claude)
    return agent.assess_risks(request.text)


@router.post("/report")
async def generate_report(request: ReportRequest):
    agent = AnalysisAgent(use_claude=request.use_claude)
    report = agent.generate_research_report(
        ticker=request.ticker,
        company_name=request.company_name,
        financial_data=request.financial_data,
        context_text=request.context_text or "",
    )
    return {"ticker": request.ticker, "report": report}


@router.post("/metrics/{ticker}", status_code=201)
async def save_metrics(
    ticker: str,
    request: MetricsRequest,
    db: AsyncSession = Depends(get_db),
):
    metric = FinancialMetric(
        ticker=ticker.upper(),
        metric_date=datetime.fromisoformat(request.metric_date),
        revenue=request.revenue,
        net_income=request.net_income,
        eps=request.eps,
        pe_ratio=request.pe_ratio,
        pb_ratio=request.pb_ratio,
        debt_to_equity=request.debt_to_equity,
        roe=request.roe,
        roa=request.roa,
        operating_margin=request.operating_margin,
        free_cash_flow=request.free_cash_flow,
        extra_metrics=request.extra_metrics,
    )
    saved = await upsert_financial_metric(db, metric)
    return {"status": "saved", "id": saved.id}


@router.get("/metrics/{ticker}/timeseries")
async def get_metrics_timeseries(
    ticker: str,
    start_date: str,
    end_date: str,
    db: AsyncSession = Depends(get_db),
):
    metrics = await get_financial_metrics_timeseries(
        db,
        ticker=ticker,
        start_date=datetime.fromisoformat(start_date),
        end_date=datetime.fromisoformat(end_date),
    )
    return [
        {
            "metric_date": m.metric_date.isoformat(),
            "revenue": m.revenue,
            "net_income": m.net_income,
            "eps": m.eps,
            "pe_ratio": m.pe_ratio,
            "roe": m.roe,
            "operating_margin": m.operating_margin,
            "free_cash_flow": m.free_cash_flow,
        }
        for m in metrics
    ]


@router.post("/statistics")
async def compute_statistics(request: StatisticsRequest):
    prices = request.prices
    if len(prices) < 2:
        raise HTTPException(status_code=400, detail="At least 2 price points required")
    returns = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
    return {
        "descriptive_stats": descriptive_stats(prices),
        "volatility_annualized": compute_volatility(prices),
        "sharpe_ratio": compute_sharpe_ratio(returns, risk_free_rate=request.risk_free_rate),
        "max_drawdown": compute_max_drawdown(prices),
        "linear_trend": linear_trend(prices),
    }


class HallucinationCheckRequest(BaseModel):
    context_chunks: list[str]
    response: str


@router.post("/hallucination-check")
async def check_hallucination(request: HallucinationCheckRequest):
    features = engineer_hallucination_reduction_features(request.context_chunks, request.response)
    return {
        "reliability_score": features["reliability_score"],
        "numeric_overlap_ratio": features["numeric_overlap_ratio"],
        "hedging_phrase_count": features["hedging_phrase_count"],
        "uncertainty_phrase_count": features["uncertainty_phrase_count"],
        "assessment": "reliable" if features["reliability_score"] >= 0.6 else "review_needed",
    }
