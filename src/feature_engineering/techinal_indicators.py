import pandas as pd
import numpy as np

def calculate_sma(df: pd.DataFrame, window: int = 20, price_col: str = "Close") -> pd.Series:
    return df.groupby(level=1)[price_col].transform(lambda x: x.rolling(window, min_periods=1).mean())

def calculate_ema(df: pd.DataFrame, window: int = 20, price_col: str = "Close") -> pd.Series:
    return df.groupby(level=1)[price_col].transform(lambda x: x.ewm(span=window, adjust=False).mean())

def calculate_rsi(df: pd.DataFrame, window: int = 14, price_col: str = "Close") -> pd.Series:
    def rsi(series):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(window=window, min_periods=1).mean()
        ma_down = down.rolling(window=window, min_periods=1).mean()
        rs = ma_up / (ma_down + 1e-10)
        return 100 - (100 / (1 + rs))
    return df.groupby(level=1)[price_col].transform(rsi)

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, price_col: str = "Close") -> pd.DataFrame:
    def macd(series):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return pd.DataFrame({'macd': macd_line, 'macd_signal': signal_line, 'macd_hist': hist}, index=series.index)
    macd_df = df.groupby(level=1)[price_col].apply(macd)
    macd_df.index = macd_df.index.droplevel(2) if macd_df.index.nlevels == 3 else macd_df.index
    return macd_df

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0, price_col: str = "Close") -> pd.DataFrame:
    def bands(series):
        sma = series.rolling(window, min_periods=1).mean()
        std = series.rolling(window, min_periods=1).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        return pd.DataFrame({'bb_upper': upper, 'bb_lower': lower, 'bb_middle': sma}, index=series.index)
    bb_df = df.groupby(level=1)[price_col].apply(bands)
    bb_df.index = bb_df.index.droplevel(2) if bb_df.index.nlevels == 3 else bb_df.index
    return bb_df

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    def atr_func(x):
        high = x['High']
        low = x['Low']
        close = x['Close']
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(window, min_periods=1).mean()
    return df.groupby(level=1)[['High', 'Low', 'Close']].apply(atr_func).droplevel(2) if df.index.nlevels == 3 else df.groupby(level=1)[['High', 'Low', 'Close']].apply(atr_func)

def calculate_obv(df: pd.DataFrame, price_col: str = "Close", volume_col: str = "Volume") -> pd.Series:
    def obv_func(x):
        direction = np.sign(x[price_col].diff().fillna(0))
        return (direction * x[volume_col]).cumsum()
    return df.groupby(level=1)[[price_col, volume_col]].apply(obv_func).droplevel(2) if df.index.nlevels == 3 else df.groupby(level=1)[[price_col, volume_col]].apply(obv_func)

def calculate_stochastic_oscillator(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    def stoch(x):
        low_min = x['Low'].rolling(window, min_periods=1).min()
        high_max = x['High'].rolling(window, min_periods=1).max()
        k = 100 * (x['Close'] - low_min) / (high_max - low_min + 1e-10)
        d = k.rolling(3, min_periods=1).mean()
        return pd.DataFrame({'stoch_k': k, 'stoch_d': d}, index=x.index)
    stoch_df = df.groupby(level=1)[['High', 'Low', 'Close']].apply(stoch)
    stoch_df.index = stoch_df.index.droplevel(2) if stoch_df.index.nlevels == 3 else stoch_df.index
    return stoch_df

def calculate_accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    def ad_func(x):
        clv = ((x['Close'] - x['Low']) - (x['High'] - x['Close'])) / (x['High'] - x['Low'] + 1e-10)
        ad = (clv * x['Volume']).cumsum()
        return ad
    return df.groupby(level=1)[['High', 'Low', 'Close', 'Volume']].apply(ad_func).droplevel(2) if df.index.nlevels == 3 else df.groupby(level=1)[['High', 'Low', 'Close', 'Volume']].apply(ad_func)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a suite of technical indicators to the input DataFrame.
    Returns a DataFrame with new columns for each indicator.
    """
    result = df.copy()
    result['sma_20'] = calculate_sma(df, 20)
    result['ema_20'] = calculate_ema(df, 20)
    result['rsi_14'] = calculate_rsi(df, 14)
    macd = calculate_macd(df)
    for col in macd.columns:
        result[col] = macd[col]
    bb = calculate_bollinger_bands(df)
    for col in bb.columns:
        result[col] = bb[col]
    result['atr_14'] = calculate_atr(df, 14)
    result['obv'] = calculate_obv(df)
    stoch = calculate_stochastic_oscillator(df)
    for col in stoch.columns:
        result[col] = stoch[col]
    result['ad_line'] = calculate_accumulation_distribution(df)
    return result