import glob
import pandas as pd
from elevate_stat import config, storage


def _slug(s: str) -> str:
    return s.lower().replace(" ", "-")


def pbp_game_paths(season):
    return sorted(glob.glob(str(storage.raw_path("play_by_play", season, "*.parquet"))))


def load_pbp(path, columns=None):
    return pd.read_parquet(path, columns=columns)


def load_shots(season, season_type, columns=None):
    p = storage.raw_path("shots", f"{season}_{_slug(season_type)}.parquet")
    return pd.read_parquet(p, columns=columns) if p.exists() else pd.DataFrame()


def load_player_season(season, season_type, measure, columns=None):
    p = storage.raw_path("player_season", f"{season}_{_slug(season_type)}_{measure.lower()}.parquet")
    return pd.read_parquet(p, columns=columns) if p.exists() else pd.DataFrame()


def load_synergy(season, season_type, grouping="offensive"):
    pattern = str(storage.raw_path("synergy", f"{season}_{_slug(season_type)}_*_{grouping}.parquet"))
    frames = [pd.read_parquet(p) for p in sorted(glob.glob(pattern))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
