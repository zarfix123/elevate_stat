import pandas as pd
from elevate_stat import storage


def test_raw_path_joins_under_raw_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    p = storage.raw_path("play_by_play", "2015-16", "0021500001.parquet")
    assert p == tmp_path / "raw" / "play_by_play" / "2015-16" / "0021500001.parquet"


def test_save_then_load_roundtrips_and_creates_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = storage.raw_path("games", "2015-16.parquet")
    assert not storage.exists(path)
    storage.save_df(df, path)
    assert storage.exists(path)
    pd.testing.assert_frame_equal(storage.load_df(path), df)


def test_save_is_atomic_leaves_no_tmp_files(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    path = storage.raw_path("games", "2015-16.parquet")
    storage.save_df(pd.DataFrame({"a": [1]}), path)
    assert path.exists()
    assert list((tmp_path / "raw").rglob("*.tmp")) == []  # no temp file left behind
