import pytest
import numpy as np
import pandas as pd

from app.analytics.statistical_analysis import (
    compute_returns,
    compute_volatility,
    compute_sharpe_ratio,
    compute_max_drawdown,
    detect_outliers_zscore,
    linear_trend,
    moving_average,
    growth_rate_yoy,
    descriptive_stats,
)
from app.analytics.feature_engineering import (
    add_momentum_features,
    add_volatility_features,
    add_fundamental_ratios,
    add_yoy_growth_features,
    engineer_hallucination_reduction_features,
)


class TestStatisticalAnalysis:
    def test_compute_returns(self):
        prices = [100.0, 110.0, 105.0, 115.0]
        returns = compute_returns(prices)
        assert len(returns) == 3
        assert abs(returns[0] - np.log(110 / 100)) < 1e-9

    def test_compute_volatility(self):
        prices = [100.0 + i for i in range(20)]
        vol = compute_volatility(prices)
        assert vol >= 0.0

    def test_compute_sharpe_ratio(self):
        returns = [0.01, 0.02, -0.01, 0.03, 0.005]
        sharpe = compute_sharpe_ratio(returns)
        assert isinstance(sharpe, float)

    def test_compute_max_drawdown(self):
        prices = [100.0, 120.0, 90.0, 110.0]
        dd = compute_max_drawdown(prices)
        assert dd < 0.0
        assert dd == pytest.approx(-0.25, rel=1e-3)

    def test_detect_outliers_zscore(self):
        # Use enough normal values so the outlier z-score clearly exceeds threshold
        values = [1.0, 2.0, 1.5, 1.8, 1.2, 1.0, 2.0, 1.5, 1.8, 1.2, 100.0]
        outliers = detect_outliers_zscore(values)
        assert len(outliers) > 0
        assert outliers[-1] == len(values) - 1

    def test_linear_trend(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = linear_trend(values)
        assert abs(trend["slope"] - 1.0) < 1e-6
        assert abs(trend["r_squared"] - 1.0) < 1e-6

    def test_moving_average(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ma = moving_average(values, window=3)
        assert ma[0] is None
        assert ma[1] is None
        assert ma[2] == pytest.approx(2.0)
        assert ma[4] == pytest.approx(4.0)

    def test_growth_rate_yoy(self):
        assert growth_rate_yoy(110.0, 100.0) == pytest.approx(10.0)
        assert growth_rate_yoy(90.0, 100.0) == pytest.approx(-10.0)
        assert growth_rate_yoy(50.0, 0.0) is None

    def test_descriptive_stats(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = descriptive_stats(values)
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["median"] == pytest.approx(3.0)


class TestFeatureEngineering:
    def _make_price_df(self, n=60):
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({"close": prices})

    def test_add_momentum_features(self):
        df = self._make_price_df()
        result = add_momentum_features(df, windows=[5, 20])
        assert "return_5d" in result.columns
        assert "sma_20d" in result.columns
        assert "price_to_sma_5d" in result.columns

    def test_add_volatility_features(self):
        df = self._make_price_df()
        result = add_volatility_features(df, windows=[10, 20])
        assert "volatility_10d" in result.columns
        assert "volatility_20d" in result.columns

    def test_add_fundamental_ratios(self):
        df = pd.DataFrame(
            {
                "revenue": [1000.0, 1100.0],
                "net_income": [100.0, 110.0],
                "total_equity": [500.0, 550.0],
                "total_assets": [2000.0, 2100.0],
                "total_debt": [200.0, 210.0],
            }
        )
        result = add_fundamental_ratios(df)
        assert "net_margin" in result.columns
        assert "roe" in result.columns
        assert result["net_margin"].iloc[0] == pytest.approx(0.1)

    def test_add_yoy_growth_features(self):
        df = pd.DataFrame(
            {"revenue": [100.0, 105.0, 110.0, 115.0, 110.0, 115.0, 120.0, 130.0]}
        )
        result = add_yoy_growth_features(df, cols=["revenue"], periods=4)
        assert "revenue_yoy_growth" in result.columns

    def test_hallucination_reduction_features(self):
        context = ["Revenue was $100 million in Q3 2023", "Net income reached $10 million"]
        response = "Based on the filing, revenue was $100 million. The company reported net income of $10 million."
        features = engineer_hallucination_reduction_features(context, response)
        assert "reliability_score" in features
        assert 0.0 <= features["reliability_score"] <= 1.0
        assert features["numeric_overlap_ratio"] > 0

    def test_hallucination_flags_uncertain_response(self):
        context = ["Revenue was $100 million"]
        response = "I think revenue was probably around $200 million, might be higher."
        features = engineer_hallucination_reduction_features(context, response)
        assert features["uncertainty_phrase_count"] > 0
