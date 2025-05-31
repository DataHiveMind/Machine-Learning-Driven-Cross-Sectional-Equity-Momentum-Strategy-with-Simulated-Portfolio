import pandas as pd
import numpy as np

def combine_signals(predictions: pd.DataFrame, config: dict, market_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Combines raw model predictions into actionable trading signals.
    Args:
        predictions: DataFrame with index (date, ticker) and columns like 'prediction' (and possibly others).
        config: Dictionary with signal generation parameters (e.g., thresholds, percentiles, risk filters).
        market_data: Optional DataFrame for risk/liquidity filtering.
    Returns:
        DataFrame with columns: ['signal', 'rank', ...] indexed by (date, ticker).
    """
    df = predictions.copy()
    signal_col = config.get("prediction_col", "prediction")
    long_pct = config.get("long_percentile", 0.2)
    short_pct = config.get("short_percentile", 0.2)
    min_volume = config.get("min_volume", None)
    min_price = config.get("min_price", None)

    # Rank assets within each date
    df["rank"] = df.groupby(level=0)[signal_col].rank(ascending=False, method="first")
    df["pct_rank"] = df.groupby(level=0)[signal_col].rank(pct=True)

    # Generate discrete signals based on percentiles
    df["signal"] = 0  # default: hold
    df.loc[df["pct_rank"] >= (1 - long_pct), "signal"] = 1   # long
    df.loc[df["pct_rank"] <= short_pct, "signal"] = -1       # short

    # Optional: risk/liquidity filtering
    if market_data is not None:
        if min_volume is not None and "Volume" in market_data.columns:
            valid = market_data["Volume"] >= min_volume
            df.loc[~valid, "signal"] = 0
        if min_price is not None and "Close" in market_data.columns:
            valid = market_data["Close"] >= min_price
            df.loc[~valid, "signal"] = 0

    return df[["signal", "rank", "pct_rank"]]