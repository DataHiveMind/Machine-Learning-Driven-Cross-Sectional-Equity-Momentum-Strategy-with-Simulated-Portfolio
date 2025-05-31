import pandas as pd
import numpy as np

def clean_market_data(
    raw_data: pd.DataFrame,
    method_missing: str = 'ffill',
    outlier_zscore: float = 5.0,
    drop_negative_prices: bool = True,
    normalize: bool = False
) -> pd.DataFrame:
    """
    Cleans raw market data for further processing.
    Args:
        raw_data: DataFrame with MultiIndex (date, ticker) and columns like Open, High, Low, Close, Adj Close, Volume.
        method_missing: Method for missing values ('ffill', 'bfill', 'drop').
        outlier_zscore: Z-score threshold for outlier detection.
        drop_negative_prices: If True, drops rows with negative prices.
        normalize: If True, standardizes price columns.
    Returns:
        Cleaned DataFrame.
    """
    df = raw_data.copy()

    # Data type conversion
    df = df.apply(pd.to_numeric, errors='coerce')
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex (date, ticker) for raw_data.")

    # Handle missing values
    if method_missing == 'ffill':
        df = df.groupby(level=1).ffill()
    elif method_missing == 'bfill':
        df = df.groupby(level=1).bfill()
    elif method_missing == 'drop':
        df = df.dropna()
    else:
        raise ValueError("Unknown method_missing option.")

    # Outlier detection and treatment (z-score method, per ticker)
    price_cols = [col for col in df.columns if 'Close' in col or 'Open' in col or 'High' in col or 'Low' in col]
    for col in price_cols:
        z = np.abs((df[col] - df[col].groupby(level=1).transform('mean')) / df[col].groupby(level=1).transform('std'))
        df.loc[z > outlier_zscore, col] = np.nan  # Set outliers to NaN for later imputation

    # Re-impute after outlier removal
    if method_missing == 'ffill':
        df = df.groupby(level=1).ffill()
    elif method_missing == 'bfill':
        df = df.groupby(level=1).bfill()

    # Drop negative prices if required
    if drop_negative_prices:
        for col in price_cols:
            df = df[df[col] >= 0]

    # Consistency checks: Remove rows with zero or negative volume
    if 'Volume' in df.columns:
        df = df[df['Volume'] > 0]

    # Data alignment: Ensure all tickers have the same dates
    all_dates = df.index.get_level_values(0).unique()
    all_tickers = df.index.get_level_values(1).unique()
    full_index = pd.MultiIndex.from_product([all_dates, all_tickers], names=['date', 'ticker'])
    df = df.reindex(full_index)
    if method_missing == 'ffill':
        df = df.groupby(level=1).ffill()
    elif method_missing == 'bfill':
        df = df.groupby(level=1).bfill()

    # Optional normalization (z-score)
    if normalize:
        for col in price_cols:
            df[col] = df.groupby(level=1)[col].transform(lambda x: (x - x.mean()) / x.std())

    return df

def main():
    # Example usage: load raw data, clean, and save
    raw = pd.read_csv('data/raw_price_data.csv', index_col=['date', 'ticker'])
    cleaned = clean_market_data(raw)
    cleaned.to_csv('data/processed/cleaned_price_data.csv')

if __name__ == "__main__":
    main()