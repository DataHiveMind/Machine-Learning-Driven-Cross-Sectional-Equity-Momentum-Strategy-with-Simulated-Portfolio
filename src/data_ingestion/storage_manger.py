import pandas as pd
from pathlib import Path
from typing import Optional

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def save_dataframe(df: pd.DataFrame, filename: str, processed: bool = False, fmt: str = "parquet"):
    """
    Save a DataFrame to disk in the specified format and directory.
    Args:
        df: DataFrame to save.
        filename: Name of the file (without extension).
        processed: If True, save to processed directory; else, raw directory.
        fmt: File format ('parquet', 'csv', 'hdf').
    """
    directory = PROCESSED_DIR if processed else RAW_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{filename}.{fmt}"

    if fmt == "parquet":
        df.to_parquet(path)
    elif fmt == "csv":
        df.to_csv(path)
    elif fmt == "hdf":
        df.to_hdf(path, key="data", mode="w")
    else:
        raise ValueError("Unsupported file format.")

def load_dataframe(filename: str, processed: bool = False, fmt: str = "parquet", **kwargs) -> pd.DataFrame:
    """
    Load a DataFrame from disk.
    Args:
        filename: Name of the file (without extension).
        processed: If True, load from processed directory; else, raw directory.
        fmt: File format ('parquet', 'csv', 'hdf').
        kwargs: Additional arguments for pandas read functions.
    Returns:
        Loaded DataFrame.
    """
    directory = PROCESSED_DIR if processed else RAW_DIR
    path = directory / f"{filename}.{fmt}"

    if fmt == "parquet":
        return pd.read_parquet(path, **kwargs)
    elif fmt == "csv":
        return pd.read_csv(path, **kwargs)
    elif fmt == "hdf":
        return pd.read_hdf(path, key="data", **kwargs)
    else:
        raise ValueError("Unsupported file format.")

def get_data_path(filename: str, processed: bool = False, fmt: str = "parquet") -> Path:
    """
    Get the full path for a data file.
    Args:
        filename: Name of the file (without extension).
        processed: If True, processed directory; else, raw directory.
        fmt: File format.
    Returns:
        Path object.
    """
    directory = PROCESSED_DIR if processed else RAW_DIR
    return directory / f"{filename}.{fmt}"