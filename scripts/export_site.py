"""Export the LATE ratings to static JSON for the web explorer (docs/)."""
import glob
import json
import os
import pandas as pd

P = "data/processed"


def main():
    late = pd.read_parquet(f"{P}/late_ratings.parquet")
    elev = pd.read_parquet(f"{P}/elevation_teammate.parquet")[["PLAYER_ID", "elevation_centrality"]]
    pairs = pd.read_parquet(f"{P}/pairs.parquet")
    mech = None
    if os.path.exists(f"{P}/mechanism.parquet"):
        mech = pd.read_parquet(f"{P}/mechanism.parquet")[["PLAYER_ID", "vol_centrality", "eff_centrality"]]

    tf = [pd.read_parquet(f, columns=["PLAYER_ID", "TEAM_ABBREVIATION"])
          for f in glob.glob("data/raw/player_season/*_base.parquet")]
    teams = pd.concat(tf, ignore_index=True)
    primary = teams.groupby("PLAYER_ID")["TEAM_ABBREVIATION"].agg(
        lambda s: s.value_counts().index[0]).rename("team")
    allteams = teams.groupby("PLAYER_ID")["TEAM_ABBREVIATION"].agg(
        lambda s: sorted(set(s))).rename("teams")

    df = (late.merge(elev, on="PLAYER_ID", how="left")
          .merge(primary, on="PLAYER_ID", how="left")
          .merge(allteams, on="PLAYER_ID", how="left"))
    if mech is not None:
        df = df.merge(mech, on="PLAYER_ID", how="left")
    df = df[df["minutes"] >= 2000].copy()

    def num(v, nd=2):
        return None if pd.isna(v) else round(float(v), nd)

    records = [{
        "id": int(r.PLAYER_ID), "name": r.name,
        "team": None if pd.isna(r.team) else r.team,
        "teams": list(r.teams) if isinstance(r.teams, list) else [],
        "min": round(float(r.minutes)),
        "late": num(r.late), "rapm": num(r.rapm),
        "elev": num(r.elevation_centrality),
        "vol": num(getattr(r, "vol_centrality", None)) if mech is not None else None,
        "eff": num(getattr(r, "eff_centrality", None)) if mech is not None else None,
    } for r in df.itertuples(index=False)]

    edges = [{"a": int(r.A), "b": int(r.B), "lift": round(float(r.lift), 2)}
             for r in pairs.itertuples(index=False)]

    os.makedirs("docs", exist_ok=True)
    json.dump(records, open("docs/players.json", "w"))
    json.dump(edges, open("docs/edges.json", "w"))
    print(f"players {len(records)} | edges {len(edges)} | mechanism {'yes' if mech is not None else 'no'}")


if __name__ == "__main__":
    main()
