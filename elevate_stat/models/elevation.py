import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

SECONDS_PER_POSS = 28.8  # ~100 team possessions per 2880s


def _player_index(stints_df):
    players = set()
    for col in ("home_lineup", "away_lineup"):
        for lu in stints_df[col]:
            players.update(lu)
    players = sorted(players)
    return {p: i for i, p in enumerate(players)}, players


def fit(stints_df, alpha=2000.0):
    """Ridge (RAPM) on stints. Design row = +1 home players, -1 away players;
    response = point margin per 100 possessions; two weightings:
      - `rapm`: possession-weighted (standard adjusted plus-minus)
      - `late`: possession x leverage weighted (the leverage-aware metric)
    Returns per-player DataFrame sorted by `late`, with `minutes`."""
    idx, players = _player_index(stints_df)
    n, m = len(stints_df), len(players)

    home_lu = stints_df["home_lineup"].to_numpy()
    away_lu = stints_df["away_lineup"].to_numpy()
    hp = stints_df["home_pts"].to_numpy(float)
    ap = stints_df["away_pts"].to_numpy(float)
    secs = stints_df["seconds"].to_numpy(float)
    lev = stints_df["leverage"].to_numpy(float)

    rows, cols, vals = [], [], []
    y = np.zeros(n)
    poss = np.zeros(n)
    minutes = np.zeros(m)
    for r in range(n):
        p = secs[r] / SECONDS_PER_POSS
        if p <= 0:
            continue
        for pid in home_lu[r]:
            rows.append(r); cols.append(idx[pid]); vals.append(1.0); minutes[idx[pid]] += secs[r] / 60
        for pid in away_lu[r]:
            rows.append(r); cols.append(idx[pid]); vals.append(-1.0); minutes[idx[pid]] += secs[r] / 60
        y[r] = 100.0 * (hp[r] - ap[r]) / p
        poss[r] = p

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n, m))
    rapm = Ridge(alpha=alpha, fit_intercept=True).fit(X, y, sample_weight=poss)
    late = Ridge(alpha=alpha, fit_intercept=True).fit(X, y, sample_weight=poss * lev)

    out = pd.DataFrame({
        "PLAYER_ID": players,
        "late": late.coef_,
        "rapm": rapm.coef_,
        "minutes": minutes,
    })
    return out.sort_values("late", ascending=False).reset_index(drop=True)
