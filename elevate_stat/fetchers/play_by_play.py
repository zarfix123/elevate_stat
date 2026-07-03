import logging

# PlayByPlayV2 is deprecated and now returns empty JSON from the NBA API
# (nba_api GitHub issue #591); V3 is the live endpoint.
from nba_api.stats.endpoints import PlayByPlayV3
from elevate_stat import storage, resilient, client as _client

log = logging.getLogger("elevate_stat.ingest")


def fetch_play_by_play(season, game_ids, *, client=_client):
    """Fetch per-game play-by-play. One flaky game never aborts the rest."""
    def _one(gid):
        path = storage.raw_path("play_by_play", season, f"{gid}.parquet")
        if storage.exists(path):
            return
        dfs = client.call(PlayByPlayV3, game_id=gid)
        storage.save_df(dfs[0], path)

    failures = resilient.for_each(game_ids, _one, label="game")
    if failures:
        log.warning("play_by_play %s: %d/%d games failed (retry on next run)",
                    season, len(failures), len(game_ids))
    return failures
