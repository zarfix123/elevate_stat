import numpy as np
import pandas as pd
from elevate_stat import build_late, data, stints
from elevate_stat.models import win_prob


def _toy_wp():
    rng = np.random.RandomState(0)
    rows = [(rng.randint(-20, 21), rng.randint(1, 2880), int(rng.rand() < 0.5)) for _ in range(200)]
    return win_prob.fit(pd.DataFrame(rows, columns=["score_diff", "seconds_remaining", "home_won"]))


def _game():
    rows = [
        (100, 1, "P1", "Made Shot", "", "PT12M00.00S", 2, ""),
        (100, 2, "P2", "Made Shot", "", "PT11M30.00S", 4, ""),
        (100, 3, "P3", "Rebound", "", "PT11M00.00S", "", ""),
        (100, 4, "P4", "Foul", "", "PT10M45.00S", "", ""),
        (100, 5, "P5", "Turnover", "", "PT10M30.00S", "", ""),
        (200, 11, "P11", "Made Shot", "", "PT10M00.00S", "", 2),
        (200, 12, "P12", "Rebound", "", "PT09M30.00S", "", ""),
        (200, 13, "P13", "Foul", "", "PT09M00.00S", "", ""),
        (200, 14, "P14", "Turnover", "", "PT08M30.00S", "", ""),
        (200, 15, "P15", "Rebound", "", "PT08M15.00S", "", ""),
        (100, 1, "P1", "Made Shot", "", "PT07M00.00S", 6, ""),
    ]
    cols = ["teamId", "personId", "playerName", "actionType", "description",
            "clock", "scoreHome", "scoreAway"]
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "actionNumber", range(1, len(df) + 1))
    df.insert(1, "period", 1)
    df["gameId"] = "G1"
    return df


def test_build_stint_table_from_mocked_game(monkeypatch):
    monkeypatch.setattr(data, "pbp_game_paths", lambda s: ["g1"])
    monkeypatch.setattr(data, "load_pbp", lambda p, columns=None: _game())
    monkeypatch.setattr(stints, "home_away_from_games", lambda s: {"G1": (100, 200)})

    table, n_games, n_clean = build_late.build_stint_table(["2015-16"], {}, _toy_wp())
    assert n_games == 1 and n_clean == 1
    assert not table.empty
    assert {"home_lineup", "away_lineup", "home_pts", "away_pts", "seconds", "leverage"} <= set(table.columns)
