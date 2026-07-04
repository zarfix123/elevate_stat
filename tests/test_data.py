import pandas as pd
from elevate_stat import data, storage


def test_load_shots_reads_the_right_file(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    storage.save_df(pd.DataFrame({"PLAYER_ID": [1, 2], "SHOT_MADE_FLAG": [1, 0]}),
                    storage.raw_path("shots", "2015-16_regular-season.parquet"))
    out = data.load_shots("2015-16", "Regular Season")
    assert list(out["PLAYER_ID"]) == [1, 2]


def test_load_shots_missing_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    assert data.load_shots("1999-00", "Regular Season").empty


def test_pbp_game_paths_lists_game_files(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    for gid in ["001", "002"]:
        storage.save_df(pd.DataFrame({"a": [1]}), storage.raw_path("play_by_play", "2015-16", f"{gid}.parquet"))
    assert len(data.pbp_game_paths("2015-16")) == 2


def test_load_synergy_concats_playtypes(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    storage.save_df(pd.DataFrame({"PLAYER_ID": [1], "PLAY_TYPE": ["Isolation"], "POSS_PCT": [0.2]}),
                    storage.raw_path("synergy", "2015-16_regular-season_isolation_offensive.parquet"))
    storage.save_df(pd.DataFrame({"PLAYER_ID": [1], "PLAY_TYPE": ["Spotup"], "POSS_PCT": [0.3]}),
                    storage.raw_path("synergy", "2015-16_regular-season_spotup_offensive.parquet"))
    out = data.load_synergy("2015-16", "Regular Season")
    assert len(out) == 2 and set(out["PLAY_TYPE"]) == {"Isolation", "Spotup"}
