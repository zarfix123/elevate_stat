import pandas as pd
from elevate_stat.fetchers import games
from elevate_stat import storage


class FakeClient:
    def __init__(self, df):
        self._df = df
        self.calls = []

    def call(self, factory, **kwargs):
        self.calls.append(kwargs)
        return [self._df]


def test_fetch_games_writes_one_file_per_season_type(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    df = pd.DataFrame({"GAME_ID": ["001", "001", "002", "002"]})
    client = FakeClient(df)
    games.fetch_games("2015-16", client=client, season_types=["Regular Season"])
    assert storage.exists(storage.raw_path("games", "2015-16_regular-season.parquet"))
    assert len(client.calls) == 1


def test_fetch_games_skips_when_file_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    df = pd.DataFrame({"GAME_ID": ["001", "001"]})
    client = FakeClient(df)
    for _ in range(2):
        games.fetch_games("2015-16", client=client, season_types=["Regular Season"])
    assert len(client.calls) == 1  # second run skipped


def test_game_ids_are_deduped(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    df = pd.DataFrame({"GAME_ID": ["001", "001", "002", "002"]})
    games.fetch_games("2015-16", client=FakeClient(df), season_types=["Regular Season"])
    assert sorted(games.game_ids("2015-16", season_types=["Regular Season"])) == ["001", "002"]
