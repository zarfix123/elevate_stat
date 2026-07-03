from nba_api.stats.endpoints import PlayByPlayV2
from elevate_stat import storage, client as _client


def fetch_play_by_play(season, game_ids, *, client=_client):
    for gid in game_ids:
        path = storage.raw_path("play_by_play", season, f"{gid}.parquet")
        if storage.exists(path):
            continue
        dfs = client.call(PlayByPlayV2, game_id=gid)
        storage.save_df(dfs[0], path)
