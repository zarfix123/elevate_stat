import pandas as pd


def compute(stints_df, min_shared_tsa=150.0):
    """Leverage-weighted WOWY teammate elevation, measured on teammate *efficiency*
    (points per true shot attempt) — not raw scoring, which is confounded by usage.
    For every pair (A, B): B's points-per-100-shot-attempts while A is on court vs off.
    Returns:
      centrality_df[PLAYER_ID, elevation_centrality, minutes]  (how much A lifts teammate efficiency)
      pairs_df[A, B, lift, shared_tsa]                          (directed A->B efficiency lift)
    Unadjusted (on/off) — thresholds + aggregation are the guardrails."""
    p_pts, p_tsa, p_secs = {}, {}, {}    # per player: lev-wtd pts, lev-wtd tsa, raw seconds
    pair_pts, pair_tsa = {}, {}          # per (A,B): B's lev-wtd pts / tsa while A on

    for row in stints_df.itertuples(index=False):
        lev = row.leverage
        pts, tsa = row.player_pts, row.player_tsa
        for lineup in (row.home_lineup, row.away_lineup):
            for b in lineup:
                bp, bt = lev * pts.get(b, 0), lev * tsa.get(b, 0)
                p_pts[b] = p_pts.get(b, 0) + bp
                p_tsa[b] = p_tsa.get(b, 0) + bt
                p_secs[b] = p_secs.get(b, 0) + row.seconds
                for a in lineup:
                    if a != b:
                        key = (a, b)
                        pair_pts[key] = pair_pts.get(key, 0) + bp
                        pair_tsa[key] = pair_tsa.get(key, 0) + bt

    lifts = []
    cent_num, cent_den = {}, {}
    for (a, b), tsa_with in pair_tsa.items():
        tsa_without = p_tsa[b] - tsa_with
        if tsa_with < min_shared_tsa or tsa_without < min_shared_tsa:
            continue
        eff_with = pair_pts[(a, b)] / tsa_with
        eff_without = (p_pts[b] - pair_pts[(a, b)]) / tsa_without
        lift = 100.0 * (eff_with - eff_without)      # pts per 100 shot attempts
        lifts.append((a, b, lift, tsa_with))
        cent_num[a] = cent_num.get(a, 0) + tsa_with * lift
        cent_den[a] = cent_den.get(a, 0) + tsa_with

    centrality_df = pd.DataFrame(
        [(a, cent_num[a] / cent_den[a], p_secs.get(a, 0) / 60) for a in cent_den],
        columns=["PLAYER_ID", "elevation_centrality", "minutes"],
    ).sort_values("elevation_centrality", ascending=False).reset_index(drop=True)
    pairs_df = pd.DataFrame(lifts, columns=["A", "B", "lift", "shared_tsa"])
    return centrality_df, pairs_df


def elevation_by_archetype(pairs_df, archetypes):
    """Per elevator A, shared-attempt-weighted mean efficiency lift onto teammates of
    each archetype. `archetypes`: dict PLAYER_ID -> archetype id."""
    df = pairs_df.copy()
    df["archetype"] = df["B"].map(archetypes)
    df = df.dropna(subset=["archetype"])
    df["w_lift"] = df["lift"] * df["shared_tsa"]
    g = df.groupby(["A", "archetype"], as_index=False).agg(
        w_lift=("w_lift", "sum"), shared_tsa=("shared_tsa", "sum"))
    g["lift"] = g["w_lift"] / g["shared_tsa"]
    return g[["A", "archetype", "lift", "shared_tsa"]]
