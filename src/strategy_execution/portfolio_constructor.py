import pandas as pd
import numpy as np

def construct_portfolio(signals: pd.DataFrame, strategy_config: dict, total_portfolio_value: float = 1_000_000) -> pd.DataFrame:
    """
    Constructs a dollar-neutral long/short portfolio based on signals and strategy config.
    Args:
        signals: DataFrame with index (date, ticker) and columns ['signal', 'rank', ...].
        strategy_config: Dict with keys like 'num_longs', 'num_shorts', 'max_position_pct', etc.
        total_portfolio_value: Total value to allocate (default $1M, can be set by backtester).
    Returns:
        DataFrame with columns: ['ticker', 'target_weight', 'target_dollar', 'direction']
    """
    # Extract config
    num_longs = strategy_config.get("num_longs", 20)
    num_shorts = strategy_config.get("num_shorts", 20)
    max_position_pct = strategy_config.get("max_position_pct", 0.05)
    min_position_pct = strategy_config.get("min_position_pct", 0.0)
    enforce_market_neutral = strategy_config.get("market_neutral", True)

    # Assume signals is for a single date (if not, user should slice before calling)
    df = signals.copy()
    if isinstance(df.index, pd.MultiIndex):
        if len(df.index.get_level_values(0).unique()) > 1:
            raise ValueError("Signals DataFrame should be for a single date.")

    # Select longs and shorts
    longs = df[df["signal"] == 1].sort_values("rank").head(num_longs)
    shorts = df[df["signal"] == -1].sort_values("rank", ascending=False).head(num_shorts)

    # Position sizing: equal weight, subject to max/min constraints
    n_longs = len(longs)
    n_shorts = len(shorts)
    if n_longs == 0 and n_shorts == 0:
        return pd.DataFrame(columns=["ticker", "target_weight", "target_dollar", "direction"])

    long_weight = min(1.0 / n_longs if n_longs > 0 else 0, max_position_pct)
    short_weight = min(1.0 / n_shorts if n_shorts > 0 else 0, max_position_pct)

    # Apply min position constraint
    if long_weight < min_position_pct:
        long_weight = min_position_pct
    if short_weight < min_position_pct:
        short_weight = min_position_pct

    # Build target positions
    longs_out = pd.DataFrame({
        "ticker": longs.index.get_level_values(-1),
        "target_weight": long_weight,
        "direction": 1
    })
    shorts_out = pd.DataFrame({
        "ticker": shorts.index.get_level_values(-1),
        "target_weight": -short_weight,
        "direction": -1
    })

    portfolio = pd.concat([longs_out, shorts_out], ignore_index=True)

    # Normalize to enforce market neutrality (sum of abs weights = 1, long = short in $)
    if enforce_market_neutral and (n_longs > 0 and n_shorts > 0):
        total_abs_weight = portfolio["target_weight"].abs().sum()
        portfolio["target_weight"] = portfolio["target_weight"] / total_abs_weight
        # Ensure sum of long weights = sum of short weights (in abs value)
        long_sum = portfolio[portfolio["direction"] == 1]["target_weight"].sum()
        short_sum = portfolio[portfolio["direction"] == -1]["target_weight"].abs().sum()
        if not np.isclose(long_sum, short_sum):
            # Scale both sides to match
            scale = min(long_sum, short_sum)
            portfolio.loc[portfolio["direction"] == 1, "target_weight"] *= scale / long_sum
            portfolio.loc[portfolio["direction"] == -1, "target_weight"] *= scale / short_sum

    # Calculate dollar allocation
    portfolio["target_dollar"] = portfolio["target_weight"] * total_portfolio_value

    # Apply max position size constraint (again, in dollar terms)
    max_dollar = max_position_pct * total_portfolio_value
    portfolio["target_dollar"] = portfolio["target_dollar"].clip(lower=-max_dollar, upper=max_dollar)

    # Final output: ticker, direction, target_weight, target_dollar
    return portfolio[["ticker", "direction", "target_weight", "target_dollar"]]