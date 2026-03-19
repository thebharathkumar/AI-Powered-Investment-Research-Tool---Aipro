from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats


def compute_returns(prices: list[float]) -> np.ndarray:
    """Compute log returns from price series."""
    arr = np.array(prices, dtype=float)
    return np.diff(np.log(arr))


def compute_volatility(prices: list[float], annualize: bool = True) -> float:
    """Compute annualized volatility from price series."""
    returns = compute_returns(prices)
    vol = float(np.std(returns, ddof=1))
    if annualize:
        vol *= np.sqrt(252)
    return vol


def compute_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252,
) -> float:
    arr = np.array(returns, dtype=float)
    excess = arr - risk_free_rate / periods_per_year
    if np.std(excess, ddof=1) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(periods_per_year))


def compute_max_drawdown(prices: list[float]) -> float:
    arr = np.array(prices, dtype=float)
    peak = np.maximum.accumulate(arr)
    drawdown = (arr - peak) / peak
    return float(np.min(drawdown))


def detect_outliers_zscore(values: list[float], threshold: float = 3.0) -> list[int]:
    arr = np.array(values, dtype=float)
    z_scores = np.abs(stats.zscore(arr))
    return list(np.where(z_scores > threshold)[0])


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(method="pearson")


def linear_trend(values: list[float]) -> dict:
    """Fit a linear trend to a time series."""
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0, "p_value": 1.0}
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value**2),
        "p_value": float(p_value),
    }


def moving_average(values: list[float], window: int) -> list[Optional[float]]:
    arr = np.array(values, dtype=float)
    result: list[Optional[float]] = [None] * (window - 1)
    for i in range(window - 1, len(arr)):
        result.append(float(np.mean(arr[i - window + 1 : i + 1])))
    return result


def growth_rate_yoy(current: float, prior: float) -> Optional[float]:
    """Compute year-over-year growth rate as a percentage.

    Uses the absolute value of *prior* as the denominator so that the sign
    of the result correctly reflects the direction of change even when the
    base period value is negative (e.g. a prior-period loss turning into a
    profit).  Returns ``None`` when *prior* is zero to avoid division by zero.
    """
    if prior == 0:
        return None
    return round((current - prior) / abs(prior) * 100, 2)


def descriptive_stats(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "skewness": float(stats.skew(arr)),
        "kurtosis": float(stats.kurtosis(arr)),
    }
