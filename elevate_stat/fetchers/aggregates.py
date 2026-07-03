from nba_api.stats.endpoints import (
    LeagueDashPlayerStats,
    LeagueDashLineups,
    SynergyPlayTypes,
)
from elevate_stat import config, storage, resilient, client as _client

MEASURE_TYPES = ["Base", "Advanced", "Scoring", "Usage"]
PLAY_TYPES = [
    "Isolation", "Transition", "PRBallHandler", "PRRollman", "Postup",
    "Spotup", "Handoff", "Cut", "OffScreen", "OffRebound", "Misc",
]


def _slug(text: str) -> str:
    return text.lower().replace(" ", "-")


def fetch_player_season(season, *, client=_client, season_types=None):
    season_types = season_types or config.SEASON_TYPES
    units = [(st, m) for st in season_types for m in MEASURE_TYPES]

    def _one(unit):
        st, measure = unit
        path = storage.raw_path("player_season", f"{season}_{_slug(st)}_{measure.lower()}.parquet")
        if storage.exists(path):
            return
        dfs = client.call(
            LeagueDashPlayerStats,
            season=season,
            season_type_all_star=st,
            measure_type_detailed_defense=measure,
        )
        storage.save_df(dfs[0], path)

    return resilient.for_each(units, _one, label="player_season")


def player_ids(season, season_type="Regular Season"):
    path = storage.raw_path("player_season", f"{season}_{_slug(season_type)}_base.parquet")
    if not storage.exists(path):
        return []
    return storage.load_df(path)["PLAYER_ID"].unique().tolist()


def fetch_lineups(season, *, client=_client, season_types=None):
    season_types = season_types or config.SEASON_TYPES

    def _one(st):
        path = storage.raw_path("lineups", f"{season}_{_slug(st)}.parquet")
        if storage.exists(path):
            return
        dfs = client.call(
            LeagueDashLineups,
            season=season,
            season_type_all_star=st,
            group_quantity=5,
            measure_type_detailed_defense="Advanced",
        )
        storage.save_df(dfs[0], path)

    return resilient.for_each(season_types, _one, label="lineups")


def fetch_synergy(season, *, client=_client, play_types=None, season_types=None):
    play_types = play_types or PLAY_TYPES
    season_types = season_types or config.SEASON_TYPES
    units = [
        (st, pt, grouping)
        for st in season_types
        for pt in play_types
        for grouping in ("offensive", "defensive")
    ]

    def _one(unit):
        st, pt, grouping = unit
        path = storage.raw_path("synergy", f"{season}_{_slug(st)}_{_slug(pt)}_{grouping}.parquet")
        if storage.exists(path):
            return
        dfs = client.call(
            SynergyPlayTypes,
            season=season,
            season_type_all_star=st,
            play_type_nullable=pt,
            type_grouping_nullable=grouping,
            player_or_team_abbreviation="P",
        )
        storage.save_df(dfs[0], path)

    return resilient.for_each(units, _one, label="synergy")
