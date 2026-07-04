import argparse
import glob
import logging
import sys
import pandas as pd
import joblib
from elevate_stat import config, build_late, viz
from elevate_stat.models import elevation_teammate as et

log = logging.getLogger("elevate_stat.elev")

PROCESSED = config.DATA_DIR / "processed"
FIGURES = PROCESSED / "figures"


def load_archetypes(processed=PROCESSED):
    """PLAYER_ID -> most-common Phase-1 playstyle archetype across seasons."""
    frames = [pd.read_parquet(f, columns=["PLAYER_ID", "top_archetype"])
              for f in glob.glob(str(processed / "playstyle" / "*.parquet"))]
    if not frames:
        return {}
    alld = pd.concat(frames, ignore_index=True)
    mode = alld.groupby("PLAYER_ID")["top_archetype"].agg(lambda s: s.value_counts().index[0])
    return mode.to_dict()


def _ids_for(names_wanted, idn):
    rev = {}
    for pid, name in idn.items():
        rev.setdefault(name, pid)
    return [rev[n] for n in names_wanted if n in rev]


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    ap = argparse.ArgumentParser(description="Build teammate-elevation ratings + figures.")
    ap.add_argument("--seasons", nargs="+", default=None)
    ap.add_argument("--min-shared-poss", type=float, default=200.0)
    ns = ap.parse_args(argv if argv is not None else sys.argv[1:])
    seasons = ns.seasons or config.SEASONS

    idn = build_late.id_fullname()
    wp = joblib.load(PROCESSED / "win_prob_model.joblib")
    table, n_games, n_clean = build_late.build_stint_table(seasons, idn, wp)
    log.info("computing teammate elevation on %d stints (%d games)", len(table), n_games)

    cent, pairs = et.compute(table, min_shared_poss=ns.min_shared_poss)
    cent["name"] = cent["PLAYER_ID"].map(idn)
    arch = et.elevation_by_archetype(pairs, load_archetypes())

    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    cent.to_parquet(PROCESSED / "elevation_teammate.parquet", index=False)
    pairs.to_parquet(PROCESSED / "pairs.parquet", index=False)

    # Figures
    big = cent[cent["minutes"] >= 5000]
    viz.centrality_bar(big, FIGURES / "top_elevators.png", names=idn, n=20)
    warriors = _ids_for(["Stephen Curry", "Kevin Durant", "Draymond Green",
                         "Klay Thompson", "Andre Iguodala"], idn)
    if len(warriors) >= 3:
        viz.who_lifts_whom(pairs, warriors, idn, FIGURES / "warriors_network.png")
    if not big.empty:
        top = big.iloc[0]
        viz.archetype_bar(arch, top["PLAYER_ID"], top["name"], FIGURES / "top_elevator_archetypes.png")

    log.info("elevation ratings + figures written (%d players, %d pairs).", len(cent), len(pairs))


if __name__ == "__main__":
    main()
