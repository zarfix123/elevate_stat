import pandas as pd
from elevate_stat import viz


def test_centrality_bar_writes_png(tmp_path):
    df = pd.DataFrame({"PLAYER_ID": [1, 2, 3], "elevation_centrality": [3.0, 2.0, 1.0],
                       "minutes": [9000, 8000, 7000]})
    out = viz.centrality_bar(df, tmp_path / "cent.png", names={1: "A", 2: "B", 3: "C"})
    assert out.exists() and out.stat().st_size > 0


def test_who_lifts_whom_writes_png(tmp_path):
    pairs = pd.DataFrame({"A": [1, 1, 2], "B": [2, 3, 3],
                          "lift": [4.0, 2.0, 1.0], "shared_poss": [300, 300, 300]})
    out = viz.who_lifts_whom(pairs, [1, 2, 3], {1: "A", 2: "B", 3: "C"}, tmp_path / "net.png")
    assert out.exists() and out.stat().st_size > 0


def test_archetype_bar_writes_png(tmp_path):
    arch = pd.DataFrame({"A": [1, 1], "archetype": [0, 1], "lift": [5.0, 1.0],
                         "shared_poss": [200, 200]})
    out = viz.archetype_bar(arch, 1, "A", tmp_path / "arch.png")
    assert out.exists() and out.stat().st_size > 0
