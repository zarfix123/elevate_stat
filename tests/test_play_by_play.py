import pandas as pd
from elevate_stat.fetchers import play_by_play as pbp
from elevate_stat import storage


class FakeClient:
    def __init__(self):
        self.calls = []

    def call(self, factory, **kwargs):
        self.calls.append(kwargs)
        return [pd.DataFrame({"EVENTNUM": [1, 2], "GAME_ID": [kwargs["game_id"]] * 2})]


def test_fetch_writes_one_file_per_game(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    pbp.fetch_play_by_play("2015-16", ["001", "002"], client=client)
    assert storage.exists(storage.raw_path("play_by_play", "2015-16", "001.parquet"))
    assert storage.exists(storage.raw_path("play_by_play", "2015-16", "002.parquet"))
    assert len(client.calls) == 2


def test_fetch_skips_existing_games(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    pbp.fetch_play_by_play("2015-16", ["001"], client=client)
    pbp.fetch_play_by_play("2015-16", ["001", "002"], client=client)
    assert len(client.calls) == 2  # 001 once, 002 once — not 3
