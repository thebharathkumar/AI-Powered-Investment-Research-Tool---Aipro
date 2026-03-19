from __future__ import annotations

import re
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer


def add_momentum_features(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: list[int] = None,
) -> pd.DataFrame:
    """Add momentum/return features over multiple windows."""
    if windows is None:
        windows = [5, 10, 20, 60]
    df = df.copy()
    for w in windows:
        df[f"return_{w}d"] = df[price_col].pct_change(w)
        df[f"sma_{w}d"] = df[price_col].rolling(window=w).mean()
        df[f"price_to_sma_{w}d"] = df[price_col] / df[f"sma_{w}d"]
    return df


def add_volatility_features(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: list[int] = None,
) -> pd.DataFrame:
    """Add rolling volatility features."""
    if windows is None:
        windows = [10, 20, 60]
    df = df.copy()
    log_returns = np.log(df[price_col] / df[price_col].shift(1))
    for w in windows:
        df[f"volatility_{w}d"] = log_returns.rolling(window=w).std() * np.sqrt(252)
    return df


def add_fundamental_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute financial ratios from fundamental columns."""
    df = df.copy()
    if "revenue" in df.columns and "net_income" in df.columns:
        df["net_margin"] = df["net_income"] / df["revenue"].replace(0, np.nan)
    if "net_income" in df.columns and "total_equity" in df.columns:
        df["roe"] = df["net_income"] / df["total_equity"].replace(0, np.nan)
    if "net_income" in df.columns and "total_assets" in df.columns:
        df["roa"] = df["net_income"] / df["total_assets"].replace(0, np.nan)
    if "total_debt" in df.columns and "total_equity" in df.columns:
        df["debt_to_equity"] = df["total_debt"] / df["total_equity"].replace(0, np.nan)
    if "ebit" in df.columns and "revenue" in df.columns:
        df["ebit_margin"] = df["ebit"] / df["revenue"].replace(0, np.nan)
    if "free_cash_flow" in df.columns and "revenue" in df.columns:
        df["fcf_margin"] = df["free_cash_flow"] / df["revenue"].replace(0, np.nan)
    return df


def add_yoy_growth_features(
    df: pd.DataFrame,
    cols: list[str],
    periods: int = 4,
) -> pd.DataFrame:
    """Add YoY growth rates (assuming quarterly data with periods=4)."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[f"{col}_yoy_growth"] = df[col].pct_change(periods) * 100
    return df


def normalize_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    method: str = "standard",
) -> tuple[pd.DataFrame, object]:
    """Normalize/standardize feature columns."""
    df = df.copy()
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(df[feature_cols])
    df[feature_cols] = scaler.fit_transform(values)
    return df, scaler


def build_feature_matrix(
    df: pd.DataFrame,
    price_col: str = "close",
    fundamental_cols: list[str] = None,
) -> pd.DataFrame:
    """Build a comprehensive feature matrix for ML modeling."""
    df = add_momentum_features(df, price_col=price_col)
    df = add_volatility_features(df, price_col=price_col)
    if fundamental_cols:
        df = add_yoy_growth_features(df, fundamental_cols)
    df = add_fundamental_ratios(df)
    return df


def engineer_hallucination_reduction_features(context_chunks: list[str], response: str) -> dict:
    """
    Compute features that help detect/reduce LLM hallucination in financial contexts.
    Returns a feature dict that can be used to score response reliability.
    """
    features: dict[str, float] = {}
    response_lower = response.lower()
    context_text = " ".join(context_chunks).lower()

    resp_numbers = re.findall(r"\b\d+(?:\.\d+)?", response)
    ctx_numbers = re.findall(r"\b\d+(?:\.\d+)?", context_text)
    resp_number_set = set(resp_numbers)
    ctx_number_set = set(ctx_numbers)

    if resp_number_set:
        overlap = len(resp_number_set & ctx_number_set) / len(resp_number_set)
    else:
        overlap = 1.0
    features["numeric_overlap_ratio"] = overlap

    hedging_phrases = [
        "based on",
        "according to",
        "as stated",
        "the document indicates",
        "the filing shows",
        "reported",
        "disclosed",
        "per the",
        "as noted",
    ]
    hedge_count = sum(1 for p in hedging_phrases if p in response_lower)
    features["hedging_phrase_count"] = float(hedge_count)

    hallucination_flags = [
        "i believe",
        "i think",
        "probably",
        "likely around",
        "estimated to be",
        "approximately",
        "roughly",
        "might be",
        "could be",
    ]
    flag_count = sum(1 for f in hallucination_flags if f in response_lower)
    features["uncertainty_phrase_count"] = float(flag_count)

    features["response_length_ratio"] = len(response) / max(len(context_text), 1)

    reliability_score = (
        0.5 * overlap
        + 0.3 * min(hedge_count / 3.0, 1.0)
        - 0.2 * min(flag_count / 5.0, 1.0)
    )
    features["reliability_score"] = max(0.0, min(1.0, reliability_score))
    return features
