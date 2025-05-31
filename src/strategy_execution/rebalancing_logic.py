import pandas as pd
import numpy as np

def should_rebalance(current_date, last_rebalance_date, frequency="weekly"):
    """
    Determines if rebalancing should occur based on frequency.
    frequency: 'daily', 'weekly', 'monthly'
    """
    if frequency == "daily":
        return True
    elif frequency == "weekly":
        return pd.to_datetime(current_date).isoweekday() == 1 or current_date != last_rebalance_date
    elif frequency == "monthly":
        return pd.to_datetime(current_date).day == 1 or current_date != last_rebalance_date
    else:
        raise ValueError("Unknown rebalancing frequency.")

def generate_rebalance_orders(
    current_holdings: dict,
    target_portfolio: pd.DataFrame,
    prices: pd.Series,
    min_trade_pct: float = 0.001,
    transaction_cost: float = 0.0
) -> list:
    """
    Generates a list of trade orders to rebalance the portfolio.
    Args:
        current_holdings: dict {ticker: shares}
        target_portfolio: DataFrame with columns ['ticker', 'direction', 'target_weight', 'target_dollar']
        prices: Series {ticker: price}
        min_trade_pct: Minimum trade size as a % of total portfolio value to avoid small trades.
        transaction_cost: Transaction cost per trade (fractional, e.g., 0.001 for 0.1%)
    Returns:
        List of orders: [{'ticker', 'action', 'quantity', 'dollar', 'transaction_cost'}]
    """
    orders = []
    target = target_portfolio.set_index("ticker")
    all_tickers = set(current_holdings.keys()).union(target.index)

    for ticker in all_tickers:
        current_shares = current_holdings.get(ticker, 0)
        price = prices.get(ticker, np.nan)
        if np.isnan(price) or price <= 0:
            continue

        target_dollar = target["target_dollar"].get(ticker, 0)
        target_shares = target_dollar / price

        diff_shares = target_shares - current_shares
        trade_dollar = diff_shares * price

        # Skip small trades
        if abs(trade_dollar) < abs(target_dollar) * min_trade_pct:
            continue

        if diff_shares > 0:
            action = "buy"
        elif diff_shares < 0:
            action = "sell"
        else:
            continue

        cost = abs(trade_dollar) * transaction_cost

        orders.append({
            "ticker": ticker,
            "action": action,
            "quantity": int(round(abs(diff_shares))),
            "dollar": abs(trade_dollar),
            "transaction_cost": cost
        })

    return orders