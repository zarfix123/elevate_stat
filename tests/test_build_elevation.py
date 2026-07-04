import pandas as pd
from elevate_stat import build_elevation


def test_load_archetypes_picks_mode_per_player(tmp_path):
    d = tmp_path / "playstyle"
    d.mkdir()
    pd.DataFrame({"PLAYER_ID": [1, 1, 2], "top_archetype": [3, 3, 5]}).to_parquet(d / "2015-16.parquet")
    pd.DataFrame({"PLAYER_ID": [1, 2], "top_archetype": [3, 5]}).to_parquet(d / "2016-17.parquet")
    m = build_elevation.load_archetypes(processed=tmp_path)
    assert m[1] == 3 and m[2] == 5


def test_ids_for_resolves_names():
    idn = {10: "Stephen Curry", 20: "Draymond Green", 30: "Nobody"}
    assert build_elevation._ids_for(["Stephen Curry", "Draymond Green", "Missing"], idn) == [10, 20]
