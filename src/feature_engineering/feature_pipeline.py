import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import yaml

from src.feature_engineering.techinal_indicators import add_technical_indicators
from src.feature_engineering.cross_sectional_features import add_cross_sectional_features

FEATURES_DIR = Path("data/features")
PROCESSED_DIR = Path("data/processed")

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def create_target_variable(df: pd.DataFrame, price_col: str = "Close", horizon: int = 1) -> pd.Series:
    """
    Example: Next-day forward return as target.
    """
    return df.groupby(level=1)[price_col].shift(-horizon) / df[price_col] - 1

def lag_features(df: pd.DataFrame, feature_cols: list, lag: int = 1) -> pd.DataFrame:
    """
    Lags features to prevent look-ahead bias.
    """
    lagged = df.groupby(level=1)[feature_cols].shift(lag)
    lagged.columns = [f"{col}_lag{lag}" for col in feature_cols]
    return lagged

def final_missing_value_handling(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    if method == "ffill":
        return df.groupby(level=1).ffill()
    elif method == "bfill":
        return df.groupby(level=1).bfill()
    elif method == "drop":
        return df.dropna()
    else:
        raise ValueError("Unknown missing value handling method.")

def scale_features(df: pd.DataFrame, feature_cols: list, scaler_type: str = "standard") -> pd.DataFrame:
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_scaled

def build_feature_set(
    cleaned_data_path: str,
    strategy_config_path: str,
    save_version: str = "v1"
):
    # Load cleaned data and config
    df = pd.read_parquet(cleaned_data_path)
    config = load_config(strategy_config_path)
    technical_features = config.get("technical_features", [])
    cross_sectional_features = config.get("cross_sectional_features", [])
    lag = config.get("feature_lag", 1)
    scaler_type = config.get("feature_scaler", "standard")
    missing_method = config.get("final_missing_method", "ffill")
    target_horizon = config.get("target_horizon", 1)
    target_price_col = config.get("target_price_col", "Close")

    # 1. Add technical indicators
    df = add_technical_indicators(df)

    # 2. Add cross-sectional features
    if cross_sectional_features:
        df = add_cross_sectional_features(df, cross_sectional_features)

    # 3. Lag features to avoid look-ahead bias
    all_feature_cols = [col for col in df.columns if col not in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    lagged = lag_features(df, all_feature_cols, lag=lag)
    df = pd.concat([df, lagged], axis=1)

    # 4. Create target variable
    df["target"] = create_target_variable(df, price_col=target_price_col, horizon=target_horizon)

    # 5. Final missing value handling
    df = final_missing_value_handling(df, method=missing_method)

    # 6. Final scaling
    lagged_feature_cols = [col for col in df.columns if col.endswith(f"_lag{lag}")]
    if lagged_feature_cols:
        df = scale_features(df, lagged_feature_cols, scaler_type=scaler_type)

    # 7. Save feature set
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FEATURES_DIR / f"features_{save_version}.parquet"
    df.to_parquet(save_path)
    print(f"Feature set saved to {save_path}")

    return df

def main():
    # Example usage
    cleaned_data_path = str(PROCESSED_DIR / "cleaned_price_data.parquet")
    strategy_config_path = "config/strategy_config.yaml"
    build_feature_set(cleaned_data_path, strategy_config_path, save_version="v1")

if __name__ == "__main__":
    main()