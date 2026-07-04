import numpy as np
import pandas as pd
from elevate_stat.models import playstyle


def _advanced(player_ids, minutes):
    n = len(player_ids)
    return pd.DataFrame({
        "PLAYER_ID": player_ids, "MIN": minutes,
        "USG_PCT": np.full(n, 0.25), "AST_PCT": np.full(n, 0.2),
        "OREB_PCT": np.full(n, 0.05), "DREB_PCT": np.full(n, 0.15),
        "TS_PCT": np.full(n, 0.57), "PACE": np.full(n, 100.0),
    })


def test_build_player_features_merges_and_fills_no_nans():
    adv = _advanced([1, 2], [1000, 1000])
    syn = pd.DataFrame({"PLAYER_ID": [1, 2], "PLAY_TYPE": ["Isolation", "Postup"], "POSS_PCT": [0.3, 0.4]})
    feats = playstyle.build_player_features(adv, syn)
    assert feats.index.tolist() == [1, 2]
    assert "pt_Isolation" in feats.columns
    assert not feats.isna().any().any()


def test_min_minutes_filter_drops_low_minute_players():
    adv = _advanced([1, 2], [1000, 100])
    feats = playstyle.build_player_features(adv, min_minutes=500)
    assert feats.index.tolist() == [1]


def test_gmm_separates_distinct_profiles():
    rng = np.random.RandomState(0)
    profiles = {
        0: [0.35, 0.50, 0.02, 0.10, 0.55, 100.0],  # high-usage playmaker
        1: [0.15, 0.10, 0.12, 0.25, 0.60, 98.0],   # low-usage rim big
        2: [0.25, 0.20, 0.05, 0.15, 0.58, 101.0],  # balanced wing
    }
    rows = []
    for pid in range(60):
        base = profiles[pid % 3]
        rows.append([pid] + [b + rng.randn() * 0.004 for b in base])
    adv = pd.DataFrame(rows, columns=["PLAYER_ID"] + playstyle.ADVANCED_FEATURES)
    adv["MIN"] = 1000

    feats = playstyle.build_player_features(adv)
    scaler, gmm = playstyle.fit(feats, n_components=3)
    out = playstyle.assign(scaler, gmm, feats).set_index("PLAYER_ID")

    # every player in true group 0 should share one archetype
    g0 = out.loc[[p for p in range(60) if p % 3 == 0], "top_archetype"]
    assert g0.nunique() == 1
    # soft memberships sum to 1
    prob_cols = [c for c in out.columns if c.startswith("arch_")]
    assert np.allclose(out[prob_cols].sum(axis=1), 1.0)
