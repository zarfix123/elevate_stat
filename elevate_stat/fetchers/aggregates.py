from nba_api.stats.endpoints import (
    LeagueDashPlayerStats,
    LeagueDashLineups,
    SynergyPlayTypes,
)
from elevate_stat import config, storage, client as _client

MEASURE_TYPES = ["Base", "Advanced", "Scoring", "Usage"]
PLAY_TYPES = [
    "Isolation", "Transition", "PRBallHandler", "PRRollman", "Postup",
    "Spotup", "Handoff", "Cut", "OffScreen", "OffRebound", "Misc",
]


def _slug(text: str) -> str:
    return text.lower().replace(" ", "-")


def fetch_player_season(season, *, client=_client):
    for measure in MEASURE_TYPES:
        path = storage.raw_path("player_season", f"{season}_{measure.lower()}.parquet")
        if storage.exists(path):
            continue
        dfs = client.call(
            LeagueDashPlayerStats,
            season=season,
            measure_type_detailed_defense=measure,
        )
        storage.save_df(dfs[0], path)


def player_ids(season):
    path = storage.raw_path("player_season", f"{season}_base.parquet")
    if not storage.exists(path):
        return []
    return storage.load_df(path)["PLAYER_ID"].unique().tolist()


def fetch_lineups(season, *, client=_client, season_types=None):
    season_types = season_types or config.SEASON_TYPES
    for st in season_types:
        path = storage.raw_path("lineups", f"{season}_{_slug(st)}.parquet")
        if storage.exists(path):
            continue
        dfs = client.call(
            LeagueDashLineups,
            season=season,
            season_type_all_star=st,
            group_quantity=5,
            measure_type_detailed_defense="Advanced",
        )
        storage.save_df(dfs[0], path)


def fetch_synergy(season, *, client=_client, play_types=None):
    play_types = play_types or PLAY_TYPES
    for pt in play_types:
        for grouping in ("offensive", "defensive"):
            path = storage.raw_path("synergy", f"{season}_{_slug(pt)}_{grouping}.parquet")
            if storage.exists(path):
                continue
            dfs = client.call(
                SynergyPlayTypes,
                season=season,
                play_type_nullable=pt,
                type_grouping_nullable=grouping,
                player_or_team_abbreviation="P",
                season_type_all_star="Regular Season",
            )
            storage.save_df(dfs[0], path)
