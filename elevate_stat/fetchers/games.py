from nba_api.stats.endpoints import LeagueGameLog
from elevate_stat import config, storage, client as _client


def _slug(season_type: str) -> str:
    return season_type.lower().replace(" ", "-")


def _path(season: str, season_type: str):
    return storage.raw_path("games", f"{season}_{_slug(season_type)}.parquet")


def fetch_games(season, *, client=_client, season_types=None):
    season_types = season_types or config.SEASON_TYPES
    for st in season_types:
        path = _path(season, st)
        if storage.exists(path):
            continue
        dfs = client.call(LeagueGameLog, season=season, season_type_all_star=st)
        storage.save_df(dfs[0], path)


def game_ids(season, *, season_types=None):
    season_types = season_types or config.SEASON_TYPES
    ids = []
    for st in season_types:
        path = _path(season, st)
        if storage.exists(path):
            ids.extend(storage.load_df(path)["GAME_ID"].tolist())
    return sorted(set(ids))
