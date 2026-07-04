import pandas as pd

SECONDS_PER_POSS = 28.8


def compute(stints_df, min_shared_poss=100.0):
    """Leverage-weighted WOWY teammate elevation. For every teammate pair (A, B),
    compare B's scoring rate (per 100 poss) while A is on court vs. off. Returns:
      centrality_df[PLAYER_ID, elevation_centrality, minutes]  (how much A lifts teammates)
      pairs_df[A, B, lift, shared_poss]                        (directed A->B lift)
    Unadjusted (on/off) — thresholds + shrinkage from aggregation are the guardrails."""
    p_pts, p_poss, p_secs = {}, {}, {}   # per player: lev-wtd pts, lev-wtd poss, raw seconds
    pair_pts, pair_poss = {}, {}         # per (A,B): B's lev-wtd pts / poss while A on

    for row in stints_df.itertuples(index=False):
        poss = row.seconds / SECONDS_PER_POSS
        wp = row.leverage * poss
        pts = row.player_pts
        for lineup in (row.home_lineup, row.away_lineup):
            for b in lineup:
                bpts = row.leverage * pts.get(b, 0)
                p_pts[b] = p_pts.get(b, 0) + bpts
                p_poss[b] = p_poss.get(b, 0) + wp
                p_secs[b] = p_secs.get(b, 0) + row.seconds
                for a in lineup:
                    if a != b:
                        key = (a, b)
                        pair_pts[key] = pair_pts.get(key, 0) + bpts
                        pair_poss[key] = pair_poss.get(key, 0) + wp

    lifts = []
    cent_num, cent_den = {}, {}
    for (a, b), poss_with in pair_poss.items():
        poss_without = p_poss[b] - poss_with
        if poss_with < min_shared_poss or poss_without < min_shared_poss:
            continue
        rate_with = 100.0 * pair_pts[(a, b)] / poss_with
        rate_without = 100.0 * (p_pts[b] - pair_pts[(a, b)]) / poss_without
        lift = rate_with - rate_without
        lifts.append((a, b, lift, poss_with))
        cent_num[a] = cent_num.get(a, 0) + poss_with * lift
        cent_den[a] = cent_den.get(a, 0) + poss_with

    centrality_df = pd.DataFrame(
        [(a, cent_num[a] / cent_den[a], p_secs.get(a, 0) / 60) for a in cent_den],
        columns=["PLAYER_ID", "elevation_centrality", "minutes"],
    ).sort_values("elevation_centrality", ascending=False).reset_index(drop=True)
    pairs_df = pd.DataFrame(lifts, columns=["A", "B", "lift", "shared_poss"])
    return centrality_df, pairs_df


def elevation_by_archetype(pairs_df, archetypes):
    """Per elevator A, shared-poss-weighted mean lift onto teammates of each archetype.
    `archetypes`: dict PLAYER_ID -> archetype id. Returns long DataFrame."""
    df = pairs_df.copy()
    df["archetype"] = df["B"].map(archetypes)
    df = df.dropna(subset=["archetype"])
    df["w_lift"] = df["lift"] * df["shared_poss"]
    g = df.groupby(["A", "archetype"], as_index=False).agg(
        w_lift=("w_lift", "sum"), shared_poss=("shared_poss", "sum"))
    g["lift"] = g["w_lift"] / g["shared_poss"]
    return g[["A", "archetype", "lift", "shared_poss"]]
