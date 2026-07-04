import argparse
import json
import logging
import sys
import numpy as np
import pandas as pd
import joblib
from elevate_stat import config, data
from elevate_stat.models import win_prob, xpps, playstyle

log = logging.getLogger("elevate_stat.build")

PROCESSED = config.DATA_DIR / "processed"
PBP_COLS = ["scoreHome", "scoreAway", "period", "clock"]
SHOT_COLS = ["PLAYER_ID", "SHOT_DISTANCE", "LOC_X", "LOC_Y",
             "SHOT_ZONE_BASIC", "SHOT_TYPE", "SHOT_MADE_FLAG"]


def build_win_prob(sample_games=3000, seed=0):
    paths = [p for s in config.SEASONS for p in data.pbp_game_paths(s)]
    rng = np.random.RandomState(seed)
    if len(paths) > sample_games:
        paths = [paths[i] for i in sorted(rng.choice(len(paths), sample_games, replace=False))]
    frames = []
    for p in paths:
        try:
            frames.append(win_prob.build_training_frame(data.load_pbp(p, columns=PBP_COLS)))
        except Exception as e:  # noqa: BLE001
            log.warning("skip pbp %s: %s", p, e)
    train = pd.concat(frames, ignore_index=True)
    model = win_prob.fit(train)
    mean_lev = float(np.mean(win_prob.leverage(
        model, train["score_diff"].to_numpy(), train["seconds_remaining"].to_numpy())))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PROCESSED / "win_prob_model.joblib")
    (PROCESSED / "win_prob_meta.json").write_text(json.dumps(
        {"n_games": len(paths), "n_states": int(len(train)), "mean_leverage": mean_lev}, indent=2))
    log.info("win_prob: %d games, %d states, mean_leverage=%.4f", len(paths), len(train), mean_lev)
    return model


def build_xpps():
    frames = []
    for s in config.SEASONS:
        for st in config.SEASON_TYPES:
            df = data.load_shots(s, st, columns=SHOT_COLS)
            if not df.empty:
                frames.append(df.assign(season=s, season_type=st))
    shots = pd.concat(frames, ignore_index=True)
    model = xpps.fit(shots)
    shots["xpps"] = xpps.expected_points(model, shots).to_numpy()
    metrics = []
    for (s, st), grp in shots.groupby(["season", "season_type"]):
        m = xpps.player_shot_metrics(grp, grp["xpps"])
        m["season"], m["season_type"] = s, st
        metrics.append(m)
    metrics = pd.concat(metrics, ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PROCESSED / "xpps_model.joblib")
    metrics.to_parquet(PROCESSED / "xpps_player_metrics.parquet", index=False)
    log.info("xpps: %d shots, %d player-season rows", len(shots), len(metrics))
    return model


def build_playstyle(n_components=8):
    feat_frames = {}
    for s in config.SEASONS:
        adv = data.load_player_season(s, "Regular Season", "Advanced")
        if adv.empty:
            continue
        syn = data.load_synergy(s, "Regular Season", "offensive")
        shots = data.load_shots(s, "Regular Season", columns=["PLAYER_ID", "SHOT_TYPE"])
        feat_frames[s] = playstyle.build_player_features(adv, syn, shots)
    pooled = pd.concat(feat_frames.values())
    cols = sorted(pooled.columns)
    scaler, gmm = playstyle.fit(pooled.reindex(columns=cols, fill_value=0.0), n_components=n_components)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "gmm": gmm, "columns": cols}, PROCESSED / "playstyle_model.joblib")
    outdir = PROCESSED / "playstyle"
    outdir.mkdir(parents=True, exist_ok=True)
    for s, feats in feat_frames.items():
        assigned = playstyle.assign(scaler, gmm, feats.reindex(columns=cols, fill_value=0.0))
        assigned["season"] = s
        assigned.to_parquet(outdir / f"{s}_regular-season.parquet", index=False)
    log.info("playstyle: %d player-seasons, %d archetypes, %d seasons",
             len(pooled), n_components, len(feat_frames))
    return scaler, gmm


def parse_args(argv):
    p = argparse.ArgumentParser(description="Fit LATE Phase 1 engines on the local dataset.")
    p.add_argument("--only", nargs="+", choices=["win_prob", "xpps", "playstyle"], default=None)
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    only = set(parse_args(argv if argv is not None else sys.argv[1:]).only or
               ["win_prob", "xpps", "playstyle"])
    if "win_prob" in only:
        build_win_prob()
    if "xpps" in only:
        build_xpps()
    if "playstyle" in only:
        build_playstyle()
    log.info("Model build complete.")


if __name__ == "__main__":
    main()
