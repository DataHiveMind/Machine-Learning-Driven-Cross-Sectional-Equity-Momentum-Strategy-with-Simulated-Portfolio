import numpy as np
import pandas as pd

def simple_return(prices: pd.Series) -> pd.Series:
    """Calculate simple (arithmetic) returns."""
    return prices.pct_change()

def log_return(prices: pd.Series) -> pd.Series:
    """Calculate logarithmic returns."""
    return np.log(prices / prices.shift(1))

def annualized_return(returns: pd.Series, freq: int = 252) -> float:
    """Calculate annualized return from periodic returns."""
    compounded = (1 + returns).prod()
    n_periods = returns.count()
    return compounded ** (freq / n_periods) - 1 if n_periods > 0 else np.nan

def historical_volatility(returns: pd.Series, freq: int = 252) -> float:
    """Calculate annualized historical volatility."""
    return returns.std(ddof=1) * np.sqrt(freq)

def rolling_volatility(returns: pd.Series, window: int = 21, freq: int = 252) -> pd.Series:
    """Calculate rolling annualized volatility."""
    return returns.rolling(window).std(ddof=1) * np.sqrt(freq)

def max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate the maximum drawdown."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    return drawdown.min()

def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return the drawdown series."""
    running_max = equity_curve.cummax()
    return (equity_curve / running_max) - 1

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, freq: int = 252) -> float:
    """Calculate the annualized Sharpe ratio."""
    excess = returns - (risk_free_rate / freq)
    return excess.mean() / excess.std(ddof=1) * np.sqrt(freq) if excess.std(ddof=1) != 0 else np.nan

def adjust_for_splits(prices: pd.Series, split_factor: pd.Series) -> pd.Series:
    """Adjust prices for stock splits."""
    return prices / split_factor.cumprod()

def fx_convert(prices: pd.Series, fx_rates: pd.Series) -> pd.Series:
    """Convert prices to a base currency using FX rates."""
    return prices * fx_rates

# Add more helpers as needed for your workflow.