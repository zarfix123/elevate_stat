import pandas as pd
from elevate_stat.models import elevation_teammate as et


def _stint(home, away, player_pts, player_tsa, seconds=120, leverage=1.0):
    return dict(home_lineup=frozenset(home), away_lineup=frozenset(away),
                home_pts=0, away_pts=0, seconds=seconds, leverage=leverage,
                player_pts=player_pts, player_tsa=player_tsa)


def _dataset():
    rows = []
    for _ in range(100):   # with player 1: teammate 2 scores 8 on 4 shots -> efficient
        rows.append(_stint([1, 2, 3, 4, 5], [11, 12, 13, 14, 15], {2: 8}, {2: 4}))
    for _ in range(100):   # without 1 (6 replaces 1): 2 scores 4 on 4 shots -> inefficient
        rows.append(_stint([2, 6, 3, 4, 5], [11, 12, 13, 14, 15], {2: 4}, {2: 4}))
    return pd.DataFrame(rows)


def test_lift_positive_when_teammate_more_efficient_with_player():
    cent, pairs = et.compute(_dataset(), min_shared_tsa=10)
    lift_1_to_2 = pairs[(pairs.A == 1) & (pairs.B == 2)]["lift"].iloc[0]
    assert lift_1_to_2 > 0
    assert cent.set_index("PLAYER_ID").loc[1, "elevation_centrality"] > 0


def test_min_shared_tsa_filters_thin_pairs():
    cent, pairs = et.compute(_dataset(), min_shared_tsa=100000)
    assert pairs.empty


def test_mechanism_volume_channel():
    # teammate 2 takes MORE shots with 1 on, same efficiency -> volume channel positive
    rows = ([_stint([1, 2, 3, 4, 5], [11, 12, 13, 14, 15], {2: 8}, {2: 4})] * 100
            + [_stint([2, 6, 3, 4, 5], [11, 12, 13, 14, 15], {2: 4}, {2: 2})] * 100)
    mech, _ = et.compute_mechanism(pd.DataFrame(rows), min_shared_tsa=10, shrink_k=0)
    m = mech.set_index("PLAYER_ID").loc[1]
    assert m["vol_centrality"] > 0
    assert abs(m["eff_centrality"]) < 1e-6


def test_mechanism_efficiency_channel():
    # teammate 2 same shots, MORE efficient with 1 on -> efficiency channel positive
    rows = ([_stint([1, 2, 3, 4, 5], [11, 12, 13, 14, 15], {2: 8}, {2: 4})] * 100
            + [_stint([2, 6, 3, 4, 5], [11, 12, 13, 14, 15], {2: 4}, {2: 4})] * 100)
    mech, _ = et.compute_mechanism(pd.DataFrame(rows), min_shared_tsa=10, shrink_k=0)
    m = mech.set_index("PLAYER_ID").loc[1]
    assert m["eff_centrality"] > 0
    assert abs(m["vol_centrality"]) < 1e-6


def test_elevation_by_archetype_reflects_ordering():
    pairs = pd.DataFrame({
        "A": [1, 1, 1], "B": [2, 3, 4],
        "lift": [10.0, 12.0, 1.0], "shared_tsa": [200.0, 200.0, 200.0],
    })
    arch = {2: 0, 3: 0, 4: 1}
    out = et.elevation_by_archetype(pairs, arch).set_index("archetype")
    assert out.loc[0, "lift"] == 11.0
    assert out.loc[1, "lift"] == 1.0
