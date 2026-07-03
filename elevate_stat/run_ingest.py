import argparse
import logging
import sys
from elevate_stat import config
from elevate_stat.fetchers import games, play_by_play, shots, aggregates, tracking_shots

log = logging.getLogger("elevate_stat.ingest")


def ingest_season(season: str) -> None:
    log.info("=== %s: games ===", season)
    games.fetch_games(season)
    gids = games.game_ids(season)

    log.info("=== %s: player-season stats ===", season)
    aggregates.fetch_player_season(season)

    log.info("=== %s: shots ===", season)
    shots.fetch_shots(season)

    log.info("=== %s: tracking shots ===", season)
    tracking_shots.fetch_tracking_shots(season)

    log.info("=== %s: play-by-play (%d games) ===", season, len(gids))
    play_by_play.fetch_play_by_play(season, gids)

    log.info("=== %s: synergy ===", season)
    aggregates.fetch_synergy(season)

    log.info("=== %s: lineups ===", season)
    aggregates.fetch_lineups(season)


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
        ingest_season(season)
    log.info("Ingest complete for %d season(s).", len(seasons))


if __name__ == "__main__":
    main()
