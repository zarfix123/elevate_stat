import pandas as pd
from elevate_stat import pbp_lineups, stints, storage
from elevate_stat.models import win_prob


def _toy_wp_model():
    import numpy as np
    rng = np.random.RandomState(0)
    rows = []
    for _ in range(400):
        diff = rng.randint(-20, 21)
        secs = rng.randint(1, 2880)
        prob = 1.0 / (1.0 + np.exp(-(diff / (1.0 + secs / 600.0))))
        rows.append((diff, secs, int(rng.rand() < prob)))
    return win_prob.fit(pd.DataFrame(rows, columns=["score_diff", "seconds_remaining", "home_won"]))


def _game():
    # 1 period: home(100) 1-5 & away(200) 11-15 start; P6 subs in for P1 mid-period.
    rows = [
        (100, 1, "P1", "Made Shot", "", "PT12M00.00S", 2, "", 2),
        (100, 2, "P2", "Made Shot", "", "PT11M30.00S", 4, "", 2),
        (100, 3, "P3", "Rebound", "", "PT11M00.00S", "", "", ""),
        (100, 4, "P4", "Foul", "", "PT10M45.00S", "", "", ""),
        (100, 5, "P5", "Turnover", "", "PT10M30.00S", "", "", ""),
        (200, 11, "P11", "Made Shot", "", "PT10M00.00S", "", 2, 2),
        (200, 12, "P12", "Rebound", "", "PT09M30.00S", "", "", ""),
        (200, 13, "P13", "Foul", "", "PT09M00.00S", "", "", ""),
        (200, 14, "P14", "Turnover", "", "PT08M30.00S", "", "", ""),
        (200, 15, "P15", "Rebound", "", "PT08M15.00S", "", "", ""),
        (100, 1, "P1", "Substitution", "SUB: P6 FOR P1", "PT08M00.00S", "", "", ""),
        (100, 6, "P6", "Made Shot", "", "PT07M00.00S", 6, "", 2),
        (100, 6, "P6", "Rebound", "", "PT06M00.00S", "", "", ""),
    ]
    cols = ["teamId", "personId", "playerName", "actionType", "description",
            "clock", "scoreHome", "scoreAway", "shotValue"]
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "actionNumber", range(1, len(df) + 1))
    df.insert(1, "period", 1)
    return df


def test_home_away_from_games(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    g = pd.DataFrame({
        "GAME_ID": ["0022300001", "0022300001"],
        "TEAM_ID": [100, 200],
        "MATCHUP": ["CLE vs. IND", "IND @ CLE"],
    })
    storage.save_df(g, storage.raw_path("games", "2023-24_regular-season.parquet"))
    m = stints.home_away_from_games("2023-24")
    assert m["0022300001"] == (100, 200)


def test_build_stints_segments_and_points():
    recon, ok = pbp_lineups.reconstruct(_game())
    assert ok
    st = stints.build_stints(recon, home_id=100, away_id=200, wp_model=_toy_wp_model())
    assert len(st) == 2
    s0, s1 = st.iloc[0], st.iloc[1]
    assert set(s0["home_lineup"]) == {1, 2, 3, 4, 5}
    assert s0["home_pts"] == 4 and s0["away_pts"] == 2         # P1+P2 baskets; P11 basket
    assert s0["seconds"] == 225                                 # 720 - 495
    assert set(s1["home_lineup"]) == {2, 3, 4, 5, 6}
    assert s1["home_pts"] == 2                                  # P6 basket
    # per-player points within each stint
    assert s0["player_pts"].get(1) == 2 and s0["player_pts"].get(2) == 2 and s0["player_pts"].get(11) == 2
    assert s1["player_pts"].get(6) == 2
