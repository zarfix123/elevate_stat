import argparse
import glob
import json
import logging
import sys
import pandas as pd
import joblib
from elevate_stat import config, data, storage, pbp_lineups, stints
from elevate_stat.models import elevation

log = logging.getLogger("elevate_stat.late")

PROCESSED = config.DATA_DIR / "processed"
ALPHA = 3000.0


def id_fullname():
    idn = {}
    for f in glob.glob(str(storage.raw_path("player_season", "*_base.parquet"))):
        d = pd.read_parquet(f, columns=["PLAYER_ID", "PLAYER_NAME"])
        idn.update(dict(zip(d["PLAYER_ID"], d["PLAYER_NAME"])))
    return idn


def build_stint_table(seasons, idn, wp_model):
    frames, n_games, n_clean = [], 0, 0
    for s in seasons:
        ha = stints.home_away_from_games(s)
        for p in data.pbp_game_paths(s):
            df = data.load_pbp(p)
            if df.empty:
                continue
            gid = str(df["gameId"].iloc[0])
            if gid not in ha:
                continue
            recon, ok = pbp_lineups.reconstruct(df, id_fullname=idn)
            n_games += 1
            n_clean += int(ok)
            home_id, away_id = ha[gid]
            st = stints.build_stints(recon, home_id, away_id, wp_model)
            if len(st):
                frames.append(st)
        log.info("stints: %s done (%d games, %d clean)", s, n_games, n_clean)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return table, n_games, n_clean


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    args = argparse.ArgumentParser(description="Build LATE ratings.")
    args.add_argument("--seasons", nargs="+", default=None)
    args.add_argument("--alpha", type=float, default=ALPHA)
    ns = args.parse_args(argv if argv is not None else sys.argv[1:])
    seasons = ns.seasons or config.SEASONS

    wp_path = PROCESSED / "win_prob_model.joblib"
    if not wp_path.exists():
        raise SystemExit("Missing Phase-1 artifact data/processed/win_prob_model.joblib — run build_models first.")
    idn = id_fullname()
    table, n_games, n_clean = build_stint_table(seasons, idn, joblib.load(wp_path))
    log.info("fitting elevation on %d stints from %d games (%.0f%% fully clean)",
             len(table), n_games, 100 * n_clean / max(n_games, 1))

    ratings = elevation.fit(table, alpha=ns.alpha)
    ratings["name"] = ratings["PLAYER_ID"].map(idn)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(PROCESSED / "late_ratings.parquet", index=False)
    (PROCESSED / "elevation_meta.json").write_text(json.dumps({
        "n_stints": int(len(table)), "n_games": n_games, "n_clean_games": n_clean,
        "alpha": ns.alpha, "n_players": int(len(ratings)),
    }, indent=2))
    log.info("LATE ratings written: %d players.", len(ratings))


if __name__ == "__main__":
    main()
