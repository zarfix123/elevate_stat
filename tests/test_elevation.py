import numpy as np
import pandas as pd
from elevate_stat.models import elevation


def _varied(seed=0, n=800):
    """Player 1 appears only in stints their team wins by 8; pool players appear in
    both winning and even stints. Lineup variation makes players identifiable."""
    rng = np.random.RandomState(seed)
    home_pool, away_pool = list(range(2, 12)), list(range(20, 30))
    rows = []
    for k in range(n):
        star = (k % 2 == 0)
        if star:
            home = frozenset([1] + list(rng.choice(home_pool, 4, replace=False)))
            hp, ap = 8, 0
        else:
            home = frozenset(rng.choice(home_pool, 5, replace=False))
            hp, ap = 4, 4
        away = frozenset(rng.choice(away_pool, 5, replace=False))
        rows.append(dict(home_lineup=home, away_lineup=away,
                         home_pts=hp, away_pts=ap, seconds=120, leverage=0.5))
    return pd.DataFrame(rows)


def test_star_gets_highest_positive_rating():
    out = elevation.fit(_varied(), alpha=50).set_index("PLAYER_ID")
    assert out.loc[1, "rapm"] > 0
    assert out.loc[1, "rapm"] > out.loc[2, "rapm"]      # star beats a pool teammate
    assert {"late", "rapm", "minutes"} <= set(out.columns)


def test_minutes_accumulate():
    out = elevation.fit(_varied(n=800), alpha=100).set_index("PLAYER_ID")
    assert out.loc[1, "minutes"] == 400 * 120 / 60      # in 400 of 800 stints -> 800 min


def test_ridge_shrinks_toward_zero_with_high_alpha():
    st = _varied()
    lo = abs(elevation.fit(st, alpha=1).set_index("PLAYER_ID").loc[1, "rapm"])
    hi = abs(elevation.fit(st, alpha=1_000_000).set_index("PLAYER_ID").loc[1, "rapm"])
    assert hi < lo
