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
    )
    assert storage.exists(storage.raw_path("tracking_shots", "2015-16", "0-2-feet-very-tight.parquet"))
    assert storage.exists(storage.raw_path("tracking_shots", "2015-16", "6plus-feet-wide-open.parquet"))
    assert len(client.calls) == 2
    assert client.calls[0]["close_def_dist_range_nullable"] == "0-2 Feet - Very Tight"


def test_fetch_skips_existing(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    for _ in range(2):
        tracking_shots.fetch_tracking_shots(
            "2015-16", client=client, def_dist_ranges=["0-2 Feet - Very Tight"],
        )
    assert len(client.calls) == 1
