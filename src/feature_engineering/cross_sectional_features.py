import pandas as pd
import numpy as np

def cross_sectional_rank(df: pd.DataFrame, feature: str) -> pd.Series:
    """Rank each asset within the cross-section for each date (1 = best)."""
    return df.groupby(level=0)[feature].rank(ascending=False, method='min')

def cross_sectional_percentile(df: pd.DataFrame, feature: str) -> pd.Series:
    """Percentile rank of each asset within the cross-section for each date."""
    return df.groupby(level=0)[feature].rank(pct=True)

def relative_strength(df: pd.DataFrame, feature: str, group_col: str = None) -> pd.Series:
    """
    Compute relative strength: asset's feature minus group (or universe) mean for each date.
    If group_col is provided, computes within group (e.g., sector).
    """
    if group_col and group_col in df.columns:
        return df[feature] - df.groupby([df.index.get_level_values(0), df[group_col]])[feature].transform('mean')
    else:
        return df[feature] - df.groupby(level=0)[feature].transform('mean')

def rolling_beta(df: pd.DataFrame, asset_col: str, benchmark_col: str, window: int = 60) -> pd.Series:
    """
    Calculate rolling beta of each asset to a benchmark.
    Assumes df has columns [asset_col, benchmark_col] and MultiIndex (date, ticker).
    """
    def beta_func(x):
        if len(x) < 2:
            return np.nan
        cov = x[asset_col].rolling(window).cov(x[benchmark_col])
        var = x[benchmark_col].rolling(window).var()
        return cov / (var + 1e-10)
    return df.groupby(level=1)[[asset_col, benchmark_col]].apply(beta_func).droplevel(2) if df.index.nlevels == 3 else df.groupby(level=1)[[asset_col, benchmark_col]].apply(beta_func)

def spread_to_median(df: pd.DataFrame, feature: str) -> pd.Series:
    """Difference between asset's feature and the median of the cross-section for each date."""
    return df[feature] - df.groupby(level=0)[feature].transform('median')

def zscore_within_cross_section(df: pd.DataFrame, feature: str) -> pd.Series:
    """Z-score normalization within each cross-section (date)."""
    mean = df.groupby(level=0)[feature].transform('mean')
    std = df.groupby(level=0)[feature].transform('std')
    return (df[feature] - mean) / (std + 1e-10)

def add_cross_sectional_features(df: pd.DataFrame, features: list, group_col: str = None) -> pd.DataFrame:
    """
    Adds a suite of cross-sectional features for the specified columns.
    Returns a DataFrame with new columns for each cross-sectional feature.
    """
    result = df.copy()
    for feat in features:
        result[f'{feat}_rank'] = cross_sectional_rank(df, feat)
        result[f'{feat}_pct'] = cross_sectional_percentile(df, feat)
        result[f'{feat}_rel_strength'] = relative_strength(df, feat, group_col)
        result[f'{feat}_spread_median'] = spread_to_median(df, feat)
        result[f'{feat}_zscore'] = zscore_within_cross_section(df, feat)
    return result