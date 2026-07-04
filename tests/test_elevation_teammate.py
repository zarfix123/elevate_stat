import pandas as pd
from elevate_stat.models import elevation_teammate as et


def _stint(home, away, player_pts, seconds=120, leverage=1.0):
    return dict(home_lineup=frozenset(home), away_lineup=frozenset(away),
                home_pts=0, away_pts=0, seconds=seconds, leverage=leverage,
                player_pts=player_pts)


def _dataset():
    rows = []
    for _ in range(100):                       # 1 & 2 together -> 2 scores 8
        rows.append(_stint([1, 2, 3, 4, 5], [11, 12, 13, 14, 15], {2: 8}))
    for _ in range(100):                       # 2 without 1 (6 replaces 1) -> 2 scores 2
        rows.append(_stint([2, 6, 3, 4, 5], [11, 12, 13, 14, 15], {2: 2}))
    return pd.DataFrame(rows)


def test_lift_positive_when_teammate_scores_more_with_player():
    cent, pairs = et.compute(_dataset(), min_shared_poss=10)
    lift_1_to_2 = pairs[(pairs.A == 1) & (pairs.B == 2)]["lift"].iloc[0]
    assert lift_1_to_2 > 0
    assert cent.set_index("PLAYER_ID").loc[1, "elevation_centrality"] > 0


def test_min_shared_poss_filters_thin_pairs():
    cent, pairs = et.compute(_dataset(), min_shared_poss=100000)
    assert pairs.empty                          # nothing clears the huge threshold


def test_elevation_by_archetype_reflects_ordering():
    pairs = pd.DataFrame({
        "A": [1, 1, 1], "B": [2, 3, 4],
        "lift": [10.0, 12.0, 1.0], "shared_poss": [200.0, 200.0, 200.0],
    })
    arch = {2: 0, 3: 0, 4: 1}                    # 2,3 are archetype 0; 4 is archetype 1
    out = et.elevation_by_archetype(pairs, arch).set_index("archetype")
    assert out.loc[0, "lift"] == 11.0           # mean of 10 & 12
    assert out.loc[1, "lift"] == 1.0
