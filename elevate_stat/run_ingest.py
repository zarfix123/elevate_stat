import argparse
import logging
import sys
from elevate_stat import config
from elevate_stat.fetchers import games, play_by_play, shots, aggregates, tracking_shots

log = logging.getLogger("elevate_stat.ingest")


def _safe(label, fn):
    """Run a stage, logging and swallowing any failure so the run continues."""
    try:
        return fn()
    except Exception as err:  # noqa: BLE001 — one stage must not abort the season
        log.warning("stage failed [%s] — %s: %s (continuing)",
                    label, type(err).__name__, str(err)[:150])
        return None


def ingest_season(season: str) -> None:
    _safe(f"{season}:games", lambda: games.fetch_games(season))
    gids = games.game_ids(season)

    _safe(f"{season}:player_season", lambda: aggregates.fetch_player_season(season))
    _safe(f"{season}:shots", lambda: shots.fetch_shots(season))
    _safe(f"{season}:tracking_shots", lambda: tracking_shots.fetch_tracking_shots(season))

    log.info("=== %s: play-by-play (%d games) ===", season, len(gids))
    _safe(f"{season}:play_by_play", lambda: play_by_play.fetch_play_by_play(season, gids))

    _safe(f"{season}:synergy", lambda: aggregates.fetch_synergy(season))
    _safe(f"{season}:lineups", lambda: aggregates.fetch_lineups(season))
    log.info("=== %s: done ===", season)


def parse_args(argv):
    p = argparse.ArgumentParser(description="Ingest nba_api data for LATE.")
    p.add_argument("--seasons", nargs="+", default=None,
                   help="Subset of seasons (default: all configured seasons).")
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = parse_args(argv if argv is not None else sys.argv[1:])
    seasons = args.seasons or config.SEASONS
    for season in seasons:
        try:
            ingest_season(season)
        except Exception as err:  # noqa: BLE001 — never let one season kill the rest
            log.error("season %s aborted — %s: %s", season, type(err).__name__, str(err)[:150])
    log.info("Ingest complete for %d season(s).", len(seasons))


if __name__ == "__main__":
    main()
