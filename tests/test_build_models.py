import numpy as np
import pandas as pd
from elevate_stat import build_models, config, data


def _fake_load_pbp(path, columns=None):
    home_win = (path == "g1")
    return pd.DataFrame({
        "scoreHome": [0, 50, 100 if home_win else 90],
        "scoreAway": [0, 48, 90 if home_win else 100],
        "period": [1, 2, 4],
        "clock": ["PT12M0.0S", "PT6M0.0S", "PT0M0.0S"],
    })


def _fake_load_shots(season, season_type, columns=None):
    rng = np.random.RandomState(1)
    n = 40
    dist = rng.choice([3, 25], size=n)
    return pd.DataFrame({
        "PLAYER_ID": rng.randint(1, 6, size=n),
        "SHOT_DISTANCE": dist, "LOC_X": 0, "LOC_Y": dist * 10,
        "SHOT_ZONE_BASIC": np.where(dist == 3, "Restricted Area", "Above the Break 3"),
        "SHOT_TYPE": np.where(dist == 3, "2PT Field Goal", "3PT Field Goal"),
        "SHOT_MADE_FLAG": rng.randint(0, 2, size=n),
    })


def _fake_load_player_season(season, season_type, measure, columns=None):
    if measure != "Advanced":
        return pd.DataFrame()
    rng = np.random.RandomState(2)
    n = 30
    return pd.DataFrame({
        "PLAYER_ID": range(1, n + 1), "MIN": 1000,
        "USG_PCT": rng.rand(n), "AST_PCT": rng.rand(n),
        "OREB_PCT": rng.rand(n) * 0.1, "DREB_PCT": rng.rand(n) * 0.3,
        "TS_PCT": 0.5 + rng.rand(n) * 0.1, "PACE": 95 + rng.rand(n) * 10,
    })


def test_build_all_writes_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SEASONS", ["2015-16"])
    monkeypatch.setattr(config, "SEASON_TYPES", ["Regular Season"])
    monkeypatch.setattr(build_models, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(data, "pbp_game_paths", lambda s: ["g1", "g2"])
    monkeypatch.setattr(data, "load_pbp", _fake_load_pbp)
    monkeypatch.setattr(data, "load_shots", _fake_load_shots)
    monkeypatch.setattr(data, "load_player_season", _fake_load_player_season)
    monkeypatch.setattr(data, "load_synergy", lambda *a, **k: pd.DataFrame())

    build_models.build_win_prob(sample_games=10)
    build_models.build_xpps()
    build_models.build_playstyle(n_components=2)

    P = tmp_path / "processed"
    for artifact in ["win_prob_model.joblib", "win_prob_meta.json",
                     "xpps_model.joblib", "xpps_player_metrics.parquet",
                     "playstyle_model.joblib", "playstyle/2015-16_regular-season.parquet"]:
        assert (P / artifact).exists(), f"missing {artifact}"
