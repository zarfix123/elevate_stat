"""Generate per-season x per-type (reg / playoffs / both) LATE data for the web
explorer -> docs/data.json. Reconstructs once into a tagged stint table, then slices."""
import glob
import json
import math
import os
import pickle
import pandas as pd
import joblib
from elevate_stat import config, data, pbp_lineups, stints, build_late
from elevate_stat.models import elevation, elevation_teammate as et

P = "data/processed"
MIN_TSA = 12          # low threshold -> rich webs, incl. short playoff samples
CACHE = os.path.expanduser("~/.cache/late_tagged_table.pkl")


def rnd(v, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), nd)


def build_tagged_table(idn, wp):
    frames = []
    for s in config.SEASONS:
        ha = stints.home_away_from_games(s)
        for path in data.pbp_game_paths(s):
            df = data.load_pbp(path)
            if df.empty:
                continue
            gid = str(df["gameId"].iloc[0])
            if gid not in ha:
                continue
            recon, _ = pbp_lineups.reconstruct(df, id_fullname=idn)
            st = stints.build_stints(recon, *ha[gid], wp)
            if len(st):
                st["season"] = s
                st["stype"] = "po" if gid[2] == "4" else "reg"
                frames.append(st)
        print(f"reconstructed {s}", flush=True)
    return pd.concat(frames, ignore_index=True)


def block(table, tmap, idn, min_minutes):
    if table.empty:
        return {"players": {}, "edges": []}
    late = elevation.fit(table, alpha=3000)
    cent, pairs = et.compute(table, min_shared_tsa=MIN_TSA)
    mech, _ = et.compute_mechanism(table, min_shared_tsa=MIN_TSA)
    df = late.merge(cent[["PLAYER_ID", "elevation_centrality"]], on="PLAYER_ID", how="left") \
             .merge(mech[["PLAYER_ID", "vol_centrality", "eff_centrality"]], on="PLAYER_ID", how="left")
    players = {}
    for r in df.itertuples(index=False):
        if r.minutes < min_minutes:
            continue
        players[int(r.PLAYER_ID)] = {
            "name": idn.get(r.PLAYER_ID, ""), "team": tmap.get(r.PLAYER_ID),
            "min": round(float(r.minutes)),
            "late": rnd(r.late), "rapm": rnd(r.rapm), "elev": rnd(r.elevation_centrality),
            "vol": rnd(r.vol_centrality), "eff": rnd(r.eff_centrality),
        }
    edges = [{"a": int(r.A), "b": int(r.B), "lift": round(float(r.lift), 2), "w": round(float(r.shared_tsa), 1)}
             for r in pairs.itertuples(index=False)]
    return {"players": players, "edges": edges}


def team_map(season):
    f = f"data/raw/player_season/{season}_regular-season_base.parquet"
    if not os.path.exists(f):
        return {}
    d = pd.read_parquet(f, columns=["PLAYER_ID", "TEAM_ABBREVIATION"])
    return dict(zip(d["PLAYER_ID"], d["TEAM_ABBREVIATION"]))


def slices(table, tmap, idn, mins):
    return {
        "both": block(table, tmap, idn, mins["both"]),
        "reg": block(table[table["stype"] == "reg"], tmap, idn, mins["reg"]),
        "po": block(table[table["stype"] == "po"], tmap, idn, mins["po"]),
    }


def main():
    idn = build_late.id_fullname()
    wp = joblib.load(f"{P}/win_prob_model.joblib")
    if os.path.exists(CACHE):
        table = pickle.load(open(CACHE, "rb"))
        print(f"loaded cached table: {len(table)} stints", flush=True)
    else:
        table = build_tagged_table(idn, wp)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        pickle.dump(table, open(CACHE, "wb"))
    print(f"tagged table: {len(table)} stints", flush=True)

    out_data = {}
    for s in config.SEASONS:
        sub = table[table["season"] == s]
        out_data[s] = slices(sub, team_map(s), idn, {"both": 150, "reg": 150, "po": 60})
        print(f"done {s}", flush=True)

    tf = [pd.read_parquet(f, columns=["PLAYER_ID", "TEAM_ABBREVIATION"])
          for f in glob.glob("data/raw/player_season/*_base.parquet")]
    teams = pd.concat(tf, ignore_index=True)
    primary = teams.groupby("PLAYER_ID")["TEAM_ABBREVIATION"].agg(lambda x: x.value_counts().index[0]).to_dict()
    out_data["all"] = slices(table, primary, idn, {"both": 1500, "reg": 1500, "po": 300})
    print("done all-time", flush=True)

    names = {str(k): v for k, v in idn.items()}
    os.makedirs("docs", exist_ok=True)
    json.dump({"seasons": list(config.SEASONS), "data": out_data, "names": names}, open("docs/data.json", "w"))
    sz = os.path.getsize("docs/data.json") / 1e6
    print(f"wrote docs/data.json ({sz:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
