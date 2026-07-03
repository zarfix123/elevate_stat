from nba_api.stats.endpoints import ShotChartDetail
from elevate_stat import config, storage, client as _client


def _slug(season_type: str) -> str:
    return season_type.lower().replace(" ", "-")


def fetch_shots(season, *, client=_client, season_types=None):
    season_types = season_types or config.SEASON_TYPES
    for st in season_types:
        path = storage.raw_path("shots", f"{season}_{_slug(st)}.parquet")
        if storage.exists(path):
            continue
        dfs = client.call(
            ShotChartDetail,
            team_id=0,
            player_id=0,
            season_nullable=season,
            season_type_all_star=st,
            context_measure_simple="FGA",
        )
        storage.save_df(dfs[0], path)
