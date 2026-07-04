import pandas as pd
from elevate_stat import storage
from elevate_stat.models import win_prob


def home_away_from_games(season):
    """gameId(str) -> (home_team_id, away_team_id). MATCHUP has 'vs.' for the home
    team, '@' for the away team."""
    frames = []
    for st in ("regular-season", "playoffs"):
        p = storage.raw_path("games", f"{season}_{st}.parquet")
        if p.exists():
            frames.append(pd.read_parquet(p, columns=["GAME_ID", "TEAM_ID", "MATCHUP"]))
    if not frames:
        return {}
    g = pd.concat(frames, ignore_index=True)
    out = {}
    for gid, grp in g.groupby("GAME_ID"):
        home = grp[grp["MATCHUP"].str.contains(" vs. ", na=False)]
        away = grp[grp["MATCHUP"].str.contains(" @ ", na=False)]
        if len(home) == 1 and len(away) == 1:
            out[str(gid)] = (home["TEAM_ID"].iloc[0], away["TEAM_ID"].iloc[0])
    return out


def build_stints(recon_df, home_id, away_id, wp_model):
    """Reconstructed pbp -> one row per stint (constant valid 5v5 within a period):
    home/away lineups, point deltas, seconds, and a leverage weight at stint start."""
    team_a = recon_df.attrs.get("team_a")
    a_is_home = (team_a == home_id)

    home_score = pd.to_numeric(recon_df["scoreHome"], errors="coerce").ffill().fillna(0).to_numpy()
    away_score = pd.to_numeric(recon_df["scoreAway"], errors="coerce").ffill().fillna(0).to_numpy()
    period = recon_df["period"].to_numpy()
    clock = recon_df["clock"].map(win_prob.parse_clock).to_numpy()
    on_a = recon_df["on_a"].to_numpy()
    on_b = recon_df["on_b"].to_numpy()
    valid = recon_df["valid"].to_numpy()
    action = recon_df["actionType"].to_numpy()
    person = recon_df["personId"].to_numpy()
    shotval = pd.to_numeric(recon_df.get("shotValue", pd.Series(2, index=recon_df.index)),
                            errors="coerce").fillna(2).to_numpy()
    desc = recon_df.get("description", pd.Series("", index=recon_df.index)).astype(str).to_numpy()
    n = len(recon_df)

    rows = []
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        p, la, lb = period[i], on_a[i], on_b[i]
        j = i
        while (j + 1 < n and valid[j + 1] and period[j + 1] == p
               and on_a[j + 1] == la and on_b[j + 1] == lb):
            j += 1

        home_lu, away_lu = (la, lb) if a_is_home else (lb, la)
        seconds = max(clock[i] - clock[j], 0.0)
        # Baseline is the score entering the stint (event i-1) so points scored ON
        # the stint's first event count for that stint's lineup.
        start_h = home_score[i - 1] if i > 0 else 0.0
        start_a = away_score[i - 1] if i > 0 else 0.0
        if seconds > 0 and len(home_lu) == 5 and len(away_lu) == 5:
            secs_remaining = win_prob.seconds_remaining(p, clock[i])
            player_pts, player_tsa = {}, {}   # points and true shot attempts (FGA + 0.44*FTA)
            for k in range(i, j + 1):
                pid, a = person[k], action[k]
                if a == "Made Shot":
                    player_pts[pid] = player_pts.get(pid, 0) + shotval[k]
                    player_tsa[pid] = player_tsa.get(pid, 0) + 1
                elif a == "Missed Shot":
                    player_tsa[pid] = player_tsa.get(pid, 0) + 1
                elif a == "Free Throw":
                    player_tsa[pid] = player_tsa.get(pid, 0) + 0.44
                    if "MISS" not in desc[k]:
                        player_pts[pid] = player_pts.get(pid, 0) + 1
            rows.append({
                "home_lineup": home_lu,
                "away_lineup": away_lu,
                "home_pts": home_score[j] - start_h,
                "away_pts": away_score[j] - start_a,
                "seconds": seconds,
                "leverage": float(win_prob.leverage(wp_model, start_h - start_a, secs_remaining)),
                "player_pts": player_pts,
                "player_tsa": player_tsa,
            })
        i = j + 1

    return pd.DataFrame(rows)
