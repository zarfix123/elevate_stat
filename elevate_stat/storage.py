import os
from pathlib import Path
import pandas as pd
from elevate_stat import config


def raw_path(*parts: str) -> Path:
    """Build a path under the raw data dir, e.g. raw_path('games', '2015-16.parquet')."""
    return config.RAW_DIR.joinpath(*parts)


def exists(path: Path) -> bool:
    return path.exists()


def save_df(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet atomically.

    Writes to a temp file then os.replace()s into place, so a crash or kill
    mid-write never leaves a partial .parquet that the skip-if-exists resume
    logic would mistake for a completed unit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_df(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
