import pandas as pd
import yfinance as yf
import time
import yaml
from typing import List, Dict, Any

def load_data_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def fetch_yfinance_data(tickers: List[str], start: str, end: str, interval: str = "1d", max_retries: int = 3, pause: float = 2.0) -> pd.DataFrame:
    """
    Fetch historical market data from Yahoo Finance using yfinance.
    Args:
        tickers: List of ticker symbols.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        interval: Data frequency (e.g., '1d', '1h').
        max_retries: Number of retries on failure.
        pause: Seconds to wait between retries.
    Returns:
        DataFrame with multi-index (date, ticker) and columns: Open, High, Low, Close, Adj Close, Volume.
    """
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers=" ".join(tickers),
                start=start,
                end=end,
                interval=interval,
                group_by='ticker',
                auto_adjust=False,
                threads=True
            )
            # If single ticker, yfinance returns a single-level column index
            if len(tickers) == 1:
                data.columns = pd.MultiIndex.from_product([tickers, data.columns])
            # Stack to long format: (date, ticker, field)
            data = data.stack(level=0).reset_index().set_index(['Date', 'level_1'])
            data.index.names = ['date', 'ticker']
            return data
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(pause)
    raise RuntimeError("Failed to fetch data after multiple retries.")

def main():
    # Load config
    config = load_data_config('config/data_config.yaml')
    tickers = config['tickers']
    start_date = config['start_date']
    end_date = config['end_date']
    interval = config.get('interval', '1d')

    # Fetch data
    raw_data = fetch_yfinance_data(tickers, start_date, end_date, interval)
    # Save raw data to CSV (optional)
    raw_data.to_csv('data/raw_price_data.csv')

if __name__ == "__main__":
    main()