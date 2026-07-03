import pandas as pd
from elevate_stat.fetchers import tracking_shots
from elevate_stat import storage


class FakeClient:
    def __init__(self):
        self.calls = []

    def call(self, factory, **kwargs):
        self.calls.append(kwargs)
        return [pd.DataFrame({"PLAYER_ID": [201939], "FGA": [120], "FG_PCT": [0.48]})]


def test_fetch_writes_one_file_per_defender_bucket(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    tracking_shots.fetch_tracking_shots(
        "2015-16", client=client,
        def_dist_ranges=["0-2 Feet - Very Tight", "6+ Feet - Wide Open"],
        season_types=["Regular Season"],
    )
    assert storage.exists(storage.raw_path("tracking_shots", "2015-16", "regular-season_0-2-feet-very-tight.parquet"))
    assert storage.exists(storage.raw_path("tracking_shots", "2015-16", "regular-season_6plus-feet-wide-open.parquet"))
    assert len(client.calls) == 2
    assert client.calls[0]["close_def_dist_range_nullable"] == "0-2 Feet - Very Tight"
    assert client.calls[0]["season_type_all_star"] == "Regular Season"


def test_fetch_covers_both_season_types(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    tracking_shots.fetch_tracking_shots(
        "2015-16", client=FakeClient(),
        def_dist_ranges=["0-2 Feet - Very Tight"],
        season_types=["Regular Season", "Playoffs"],
    )
    assert storage.exists(storage.raw_path("tracking_shots", "2015-16", "regular-season_0-2-feet-very-tight.parquet"))
    assert storage.exists(storage.raw_path("tracking_shots", "2015-16", "playoffs_0-2-feet-very-tight.parquet"))


def test_fetch_skips_existing(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    for _ in range(2):
        tracking_shots.fetch_tracking_shots(
            "2015-16", client=client, def_dist_ranges=["0-2 Feet - Very Tight"],
            season_types=["Regular Season"],
        )
    assert len(client.calls) == 1
