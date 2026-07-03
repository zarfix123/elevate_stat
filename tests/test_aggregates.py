import pandas as pd
from elevate_stat.fetchers import aggregates
from elevate_stat import storage


class FakeClient:
    def __init__(self):
        self.calls = []

    def call(self, factory, **kwargs):
        self.calls.append((factory.__name__, kwargs))
        return [pd.DataFrame({"PLAYER_ID": [201939, 2544], "PTS": [30, 27]})]


def test_player_stats_writes_per_measure_and_exposes_player_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    aggregates.fetch_player_season("2015-16", client=client)
    assert storage.exists(storage.raw_path("player_season", "2015-16_base.parquet"))
    assert storage.exists(storage.raw_path("player_season", "2015-16_advanced.parquet"))
    assert sorted(aggregates.player_ids("2015-16")) == [2544, 201939]


def test_synergy_and_lineups_write_files(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    aggregates.fetch_lineups("2015-16", client=client, season_types=["Regular Season"])
    aggregates.fetch_synergy("2015-16", client=client, play_types=["Isolation"])
    assert storage.exists(storage.raw_path("lineups", "2015-16_regular-season.parquet"))
    assert storage.exists(storage.raw_path("synergy", "2015-16_isolation_offensive.parquet"))
    assert storage.exists(storage.raw_path("synergy", "2015-16_isolation_defensive.parquet"))
