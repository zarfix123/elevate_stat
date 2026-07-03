from nba_api.stats.endpoints import LeagueDashPlayerPtShot
from elevate_stat import storage, client as _client

# Closest-defender distance buckets — the key tracking context for the xPPS
# baseline (design §4.2). Pulled league-wide, one call per bucket per season,
# so all players come back together (vs. ~500 per-player calls).
DEF_DIST_RANGES = [
    "0-2 Feet - Very Tight",
    "2-4 Feet - Tight",
    "4-6 Feet - Open",
    "6+ Feet - Wide Open",
]


def _slug(text: str) -> str:
    return text.lower().replace("+", "plus").replace(" - ", "-").replace(" ", "-")


def fetch_tracking_shots(season, *, client=_client, def_dist_ranges=None):
    """League-wide per-player tracking shooting, split by closest-defender distance.

    Other split families (catch-and-shoot via general_range, dribbles, touch time)
    can be added later the same way — this covers the primary xPPS context for v1.
    """
    ranges = def_dist_ranges or DEF_DIST_RANGES
    for rng in ranges:
        path = storage.raw_path("tracking_shots", season, f"{_slug(rng)}.parquet")
        if storage.exists(path):
            continue
        dfs = client.call(
            LeagueDashPlayerPtShot,
            season=season,
            close_def_dist_range_nullable=rng,
        )
        storage.save_df(dfs[0], path)
