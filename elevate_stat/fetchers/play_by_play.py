# PlayByPlayV2 is deprecated and now returns empty JSON from the NBA API
# (nba_api GitHub issue #591); V3 is the live endpoint.
from nba_api.stats.endpoints import PlayByPlayV3
from elevate_stat import storage, client as _client


def fetch_play_by_play(season, game_ids, *, client=_client):
    for gid in game_ids:
        path = storage.raw_path("play_by_play", season, f"{gid}.parquet")
        if storage.exists(path):
            continue
        dfs = client.call(PlayByPlayV3, game_id=gid)
        storage.save_df(dfs[0], path)
