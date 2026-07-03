from pathlib import Path
import pandas as pd
from elevate_stat import config


def raw_path(*parts: str) -> Path:
    """Build a path under the raw data dir, e.g. raw_path('games', '2015-16.parquet')."""
    return config.RAW_DIR.joinpath(*parts)


def exists(path: Path) -> bool:
    return path.exists()


def save_df(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_df(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
