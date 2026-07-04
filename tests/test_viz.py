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


def test_clutch_scatter_writes_png(tmp_path):
    df = pd.DataFrame({"PLAYER_ID": [1, 2, 3], "rapm": [5.0, 3.0, 1.0],
                       "late": [6.0, 2.5, 1.5], "minutes": [20000, 15000, 13000]})
    out = viz.clutch_scatter(df, tmp_path / "clutch.png", names={1: "A", 2: "B", 3: "C"},
                             min_minutes=10000)
    assert out.exists() and out.stat().st_size > 0


def test_mechanism_map_writes_png(tmp_path):
    df = pd.DataFrame({"PLAYER_ID": [1, 2], "vol_centrality": [3.0, 1.0],
                       "eff_centrality": [1.0, 4.0], "minutes": [15000, 15000]})
    out = viz.mechanism_map(df, tmp_path / "mech.png", names={1: "A", 2: "B"})
    assert out.exists() and out.stat().st_size > 0


def test_trajectory_writes_png(tmp_path):
    df = pd.DataFrame({
        "PLAYER_ID": [1, 1, 2], "season": ["2015-16", "2016-17", "2015-16"],
        "season_type": ["Regular Season"] * 3, "points_above_expected": [50.0, 80.0, 20.0],
    })
    out = viz.trajectory(df, [1, 2], {1: "A", 2: "B"}, tmp_path / "traj.png")
    assert out.exists() and out.stat().st_size > 0


def test_archetype_bar_writes_png(tmp_path):
    arch = pd.DataFrame({"A": [1, 1], "archetype": [0, 1], "lift": [5.0, 1.0],
                         "shared_poss": [200, 200]})
    out = viz.archetype_bar(arch, 1, "A", tmp_path / "arch.png")
    assert out.exists() and out.stat().st_size > 0
