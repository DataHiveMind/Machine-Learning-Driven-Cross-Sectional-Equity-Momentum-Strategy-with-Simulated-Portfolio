import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_metrics(returns: pd.Series, benchmark_returns: pd.Series = None, risk_free_rate: float = 0.0, freq: int = 252) -> dict:
    """
    Calculate performance and risk metrics for a strategy.
    Args:
        returns: pd.Series of strategy daily returns.
        benchmark_returns: pd.Series of benchmark daily returns (optional).
        risk_free_rate: Annualized risk-free rate (as decimal).
        freq: Number of trading periods per year (default 252 for daily).
    Returns:
        Dictionary of metrics.
    """
    metrics = {}

    # Return Metrics
    cumulative_return = (1 + returns).prod() - 1
    annualized_return = (1 + cumulative_return) ** (freq / len(returns)) - 1
    avg_daily_return = returns.mean()
    avg_monthly_return = returns.resample('M').apply(lambda x: (1 + x).prod() - 1).mean() if hasattr(returns.index, 'freq') else np.nan

    # Risk Metrics
    annualized_volatility = returns.std(ddof=1) * np.sqrt(freq)
    drawdown = (1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1
    max_drawdown = drawdown.min()
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # Value at Risk (VaR) and Conditional VaR (CVaR)
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()

    # Risk-Adjusted Returns
    excess_returns = returns - (risk_free_rate / freq)
    sharpe_ratio = excess_returns.mean() / excess_returns.std(ddof=1) * np.sqrt(freq) if excess_returns.std(ddof=1) != 0 else np.nan
    downside_returns = returns[returns < 0]
    sortino_ratio = excess_returns.mean() / (downside_returns.std(ddof=1) * np.sqrt(freq)) if downside_returns.std(ddof=1) != 0 else np.nan

    # Alpha & Beta (if benchmark provided)
    alpha, beta = np.nan, np.nan
    correlation = np.nan
    if benchmark_returns is not None:
        aligned = returns.align(benchmark_returns, join='inner')
        if len(aligned[0]) > 1:
            X = aligned[1] - (risk_free_rate / freq)
            Y = aligned[0] - (risk_free_rate / freq)
            beta = np.cov(Y, X)[0, 1] / np.var(X)
            alpha = (Y.mean() - beta * X.mean()) * freq
            correlation = Y.corr(X)

    metrics.update({
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'avg_daily_return': avg_daily_return,
        'avg_monthly_return': avg_monthly_return,
        'annualized_volatility': annualized_volatility,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'VaR_95': var_95,
        'CVaR_95': cvar_95,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'alpha': alpha,
        'beta': beta,
        'market_correlation': correlation,
    })

    return metrics

# Optional: Trade-level analysis (stub)
def trade_analysis(trade_log: list) -> dict:
    """
    Analyze trade-level statistics.
    Args:
        trade_log: List of trade records (dicts).
    Returns:
        Dictionary of trade metrics.
    """
    if not trade_log:
        return {}

    profits = [trade['profit'] for trade in trade_log]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]
    win_rate = len(wins) / len(profits) if profits else np.nan
    profit_factor = sum(wins) / abs(sum(losses)) if losses else np.nan
    avg_profit = np.mean(profits) if profits else np.nan

    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_profit_per_trade': avg_profit,
        'num_trades': len(profits)
    }