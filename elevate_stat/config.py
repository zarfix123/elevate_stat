from pathlib import Path


def _season_str(start_year: int) -> str:
    """2015 -> '2015-16'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


# 2015-16 is the first season with Synergy playtypes (see design spec §6).
_FIRST_START_YEAR = 2015
_LAST_START_YEAR = 2025  # 2025-26 season, completed June 2026
SEASONS = [_season_str(y) for y in range(_FIRST_START_YEAR, _LAST_START_YEAR + 1)]

SEASON_TYPES = ["Regular Season", "Playoffs"]

# Storage
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"

# Network tuning (stats.nba.com is rate-limited and flaky)
REQUEST_DELAY = 0.6   # seconds of polite delay before each call
MAX_RETRIES = 5       # attempts per call before giving up
TIMEOUT = 60          # per-request timeout in seconds
