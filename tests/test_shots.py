import pandas as pd
from elevate_stat.fetchers import shots
from elevate_stat import storage


class FakeClient:
    def __init__(self):
        self.calls = []

    def call(self, factory, **kwargs):
        self.calls.append(kwargs)
        return [pd.DataFrame({"LOC_X": [1], "LOC_Y": [2], "SHOT_MADE_FLAG": [1]})]


def test_fetch_shots_uses_all_player_all_team_and_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    shots.fetch_shots("2015-16", client=client, season_types=["Regular Season"])
    assert storage.exists(storage.raw_path("shots", "2015-16_regular-season.parquet"))
    assert client.calls[0]["player_id"] == 0
    assert client.calls[0]["team_id"] == 0
    assert client.calls[0]["context_measure_simple"] == "FGA"


def test_fetch_shots_skips_existing(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    for _ in range(2):
        shots.fetch_shots("2015-16", client=client, season_types=["Regular Season"])
    assert len(client.calls) == 1
