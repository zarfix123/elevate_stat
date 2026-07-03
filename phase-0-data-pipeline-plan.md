# Phase 0 — Data Ingestion Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, rate-limited pipeline that pulls all `nba_api` data needed for the LATE metric (2015–16 → present) to local parquet files, then launch the full multi-hour scrape as a background job.

**Architecture:** A small Python package (`elevate_stat`) with a thin retry/rate-limit wrapper around `nba_api`, a file-existence-based checkpoint system (each fetched unit is one parquet file; if the file exists, the unit is skipped — so any interrupted run resumes for free), and one focused fetcher module per endpoint family. A single orchestrator CLI (`run_ingest.py`) walks every season × endpoint in dependency order and is safe to re-run any number of times.

**Tech Stack:** Python 3.10+, `nba_api`, `pandas`, `pyarrow` (parquet), `pytest`.

---

## File Structure

```
elevate_stat/
  __init__.py
  config.py              # season list, paths, rate-limit constants
  storage.py             # path conventions + parquet read/write/exists
  client.py              # rate-limited retry wrapper around nba_api endpoints
  fetchers/
    __init__.py
    games.py             # per-season game index (LeagueGameLog)
    play_by_play.py      # per-game play-by-play (PlayByPlayV2)
    shots.py             # per-season all-player shots (ShotChartDetail trick)
    aggregates.py        # player-season stats, Synergy, lineups (season-level)
    shot_logs.py         # per-player tracking shot logs (PlayerDashPtShotLog)
  run_ingest.py          # orchestrator CLI
tests/
  test_config.py
  test_storage.py
  test_client.py
  test_games.py
  test_play_by_play.py
  test_shots.py
  test_aggregates.py
  test_shot_logs.py
  test_run_ingest.py
requirements.txt
pytest.ini
```

**Responsibilities:**
- `config` — the only place seasons/paths/tuning constants live (DRY).
- `storage` — the only place that knows the on-disk layout; everything else asks it for paths.
- `client` — the only place that talks to the network; handles delay/retry so fetchers stay simple.
- `fetchers/*` — one endpoint family each, all following the same "skip if exists, else fetch + save" shape.
- `run_ingest` — wiring only; no fetch logic of its own.

---

## Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `elevate_stat/__init__.py`, `elevate_stat/fetchers/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
nba_api==1.4.1
pandas>=2.0
pyarrow>=14.0
pytest>=8.0
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 3: Create empty package init files**

Create `elevate_stat/__init__.py` containing:

```python
"""LATE — data ingestion pipeline."""
```

Create `elevate_stat/fetchers/__init__.py` as an empty file.

- [ ] **Step 4: Create and activate a virtualenv, install deps**

Run:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
Expected: installs complete without error; `.venv/` is already covered by `.gitignore`.

- [ ] **Step 5: Verify pytest runs (no tests yet)**

Run: `.venv/bin/pytest`
Expected: `no tests ran` (exit code 5) — confirms pytest is wired up.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini elevate_stat/
git commit -m "chore: scaffold elevate_stat package and test harness"
```

---

## Task 2: Config module

**Files:**
- Create: `elevate_stat/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import re
from elevate_stat import config


def test_seasons_span_2015_16_through_2025_26():
    assert config.SEASONS[0] == "2015-16"
    assert config.SEASONS[-1] == "2025-26"
    assert len(config.SEASONS) == 11


def test_every_season_matches_nba_format():
    for season in config.SEASONS:
        assert re.fullmatch(r"\d{4}-\d{2}", season), season


def test_tuning_constants_present():
    assert config.REQUEST_DELAY > 0
    assert config.MAX_RETRIES >= 1
    assert config.TIMEOUT >= 30
    assert config.SEASON_TYPES == ["Regular Season", "Playoffs"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: elevate_stat.config`

- [ ] **Step 3: Write minimal implementation**

```python
# elevate_stat/config.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add elevate_stat/config.py tests/test_config.py
git commit -m "feat: season list and pipeline tuning config"
```

---

## Task 3: Storage layer

**Files:**
- Create: `elevate_stat/storage.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage.py
import pandas as pd
from elevate_stat import storage


def test_raw_path_joins_under_raw_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    p = storage.raw_path("play_by_play", "2015-16", "0021500001.parquet")
    assert p == tmp_path / "raw" / "play_by_play" / "2015-16" / "0021500001.parquet"


def test_save_then_load_roundtrips_and_creates_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = storage.raw_path("games", "2015-16.parquet")
    assert not storage.exists(path)
    storage.save_df(df, path)
    assert storage.exists(path)
    pd.testing.assert_frame_equal(storage.load_df(path), df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_storage.py -v`
Expected: FAIL — `ModuleNotFoundError: elevate_stat.storage`

- [ ] **Step 3: Write minimal implementation**

```python
# elevate_stat/storage.py
from pathlib import Path
import pandas as pd
from elevate_stat import config


def raw_path(*parts: str) -> Path:
    """Build a path under the raw data dir, e.g. raw_path('games', '2015-16.parquet')."""
    return config.RAW_DIR.joinpath(*parts)


def exists(path: Path) -> bool:
    return path.exists()


def save_df(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_df(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_storage.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add elevate_stat/storage.py tests/test_storage.py
git commit -m "feat: parquet storage layer with path conventions"
```

---

## Task 4: Rate-limited retry client

**Files:**
- Create: `elevate_stat/client.py`
- Test: `tests/test_client.py`

The client wraps an `nba_api` endpoint class. It sleeps a polite delay before each call, instantiates the endpoint with a timeout, and retries with exponential backoff on any exception (network flakiness is the norm here). `sleep` is injectable so tests run instantly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_client.py
import pytest
from elevate_stat.client import call


class _FakeEndpoint:
    """Fails `fail_times` then returns a sentinel from get_data_frames()."""
    instances = 0

    def __init__(self, fail_times=0, **kwargs):
        type(self).instances += 1
        self._fail_times = fail_times
        self.kwargs = kwargs

    def get_data_frames(self):
        if type(self).instances <= self._fail_times:
            raise ConnectionError("boom")
        return ["OK"]


def _make_fake(fail_times):
    _FakeEndpoint.instances = 0

    def factory(**kwargs):
        return _FakeEndpoint(fail_times=fail_times, **kwargs)

    return factory


def test_call_returns_data_frames_on_success():
    result = call(_make_fake(0), sleep=lambda _: None, game_id="X")
    assert result == ["OK"]


def test_call_retries_then_succeeds():
    result = call(_make_fake(2), sleep=lambda _: None, max_retries=5)
    assert result == ["OK"]
    assert _FakeEndpoint.instances == 3  # 2 failures + 1 success


def test_call_raises_after_exhausting_retries():
    with pytest.raises(RuntimeError):
        call(_make_fake(99), sleep=lambda _: None, max_retries=3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_client.py -v`
Expected: FAIL — `ModuleNotFoundError: elevate_stat.client`

- [ ] **Step 3: Write minimal implementation**

```python
# elevate_stat/client.py
import time
from typing import Callable
from elevate_stat import config


def call(
    endpoint_factory: Callable,
    *,
    delay: float | None = None,
    max_retries: int | None = None,
    timeout: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs,
):
    """Call an nba_api endpoint with a polite delay + retry/backoff.

    `endpoint_factory(**kwargs)` returns an object with `.get_data_frames()`.
    Real usage passes an nba_api endpoint class (e.g. PlayByPlayV2).
    """
    delay = config.REQUEST_DELAY if delay is None else delay
    max_retries = config.MAX_RETRIES if max_retries is None else max_retries
    timeout = config.TIMEOUT if timeout is None else timeout

    last_err = None
    for attempt in range(max_retries):
        sleep(delay)  # be polite before every attempt
        try:
            endpoint = endpoint_factory(timeout=timeout, **kwargs)
            return endpoint.get_data_frames()
        except Exception as err:  # noqa: BLE001 — network layer is genuinely unpredictable
            last_err = err
            backoff = min(2 ** attempt, 30)
            sleep(backoff)
    raise RuntimeError(f"call failed after {max_retries} attempts: {last_err}") from last_err
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_client.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add elevate_stat/client.py tests/test_client.py
git commit -m "feat: rate-limited retry client for nba_api"
```

---

## Task 5: Games fetcher

**Files:**
- Create: `elevate_stat/fetchers/games.py`
- Test: `tests/test_games.py`

Pulls the per-season game index (one parquet per season+type) and exposes deduplicated game IDs. `LeagueGameLog` returns two rows per game (one per team), so `game_ids` must dedupe.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_games.py
import pandas as pd
from elevate_stat.fetchers import games
from elevate_stat import storage


class FakeClient:
    def __init__(self, df):
        self._df = df
        self.calls = []

    def call(self, factory, **kwargs):
        self.calls.append(kwargs)
        return [self._df]


def test_fetch_games_writes_one_file_per_season_type(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    df = pd.DataFrame({"GAME_ID": ["001", "001", "002", "002"]})
    client = FakeClient(df)
    games.fetch_games("2015-16", client=client, season_types=["Regular Season"])
    assert storage.exists(storage.raw_path("games", "2015-16_regular-season.parquet"))
    assert len(client.calls) == 1


def test_fetch_games_skips_when_file_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    df = pd.DataFrame({"GAME_ID": ["001", "001"]})
    client = FakeClient(df)
    for _ in range(2):
        games.fetch_games("2015-16", client=client, season_types=["Regular Season"])
    assert len(client.calls) == 1  # second run skipped


def test_game_ids_are_deduped(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    df = pd.DataFrame({"GAME_ID": ["001", "001", "002", "002"]})
    games.fetch_games("2015-16", client=FakeClient(df), season_types=["Regular Season"])
    assert sorted(games.game_ids("2015-16", season_types=["Regular Season"])) == ["001", "002"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_games.py -v`
Expected: FAIL — `ModuleNotFoundError: elevate_stat.fetchers.games`

- [ ] **Step 3: Write minimal implementation**

```python
# elevate_stat/fetchers/games.py
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
```

> Note: `client.call` is the module-level function from Task 4; the `FakeClient` in tests provides a `.call` method with the same signature, which is why fetchers accept `client=` by dependency injection.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_games.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add elevate_stat/fetchers/games.py tests/test_games.py
git commit -m "feat: per-season game index fetcher"
```

---

## Task 6: Play-by-play fetcher

**Files:**
- Create: `elevate_stat/fetchers/play_by_play.py`
- Test: `tests/test_play_by_play.py`

One parquet per game under `play_by_play/{season}/{game_id}.parquet`. This is the largest dataset. Skip-if-exists gives free resume across the ~14.5k games.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_play_by_play.py
import pandas as pd
from elevate_stat.fetchers import play_by_play as pbp
from elevate_stat import storage


class FakeClient:
    def __init__(self):
        self.calls = []

    def call(self, factory, **kwargs):
        self.calls.append(kwargs)
        return [pd.DataFrame({"EVENTNUM": [1, 2], "GAME_ID": [kwargs["game_id"]] * 2})]


def test_fetch_writes_one_file_per_game(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    pbp.fetch_play_by_play("2015-16", ["001", "002"], client=client)
    assert storage.exists(storage.raw_path("play_by_play", "2015-16", "001.parquet"))
    assert storage.exists(storage.raw_path("play_by_play", "2015-16", "002.parquet"))
    assert len(client.calls) == 2


def test_fetch_skips_existing_games(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    pbp.fetch_play_by_play("2015-16", ["001"], client=client)
    pbp.fetch_play_by_play("2015-16", ["001", "002"], client=client)
    assert len(client.calls) == 2  # 001 once, 002 once — not 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_play_by_play.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# elevate_stat/fetchers/play_by_play.py
from nba_api.stats.endpoints import PlayByPlayV2
from elevate_stat import storage, client as _client


def fetch_play_by_play(season, game_ids, *, client=_client):
    for gid in game_ids:
        path = storage.raw_path("play_by_play", season, f"{gid}.parquet")
        if storage.exists(path):
            continue
        dfs = client.call(PlayByPlayV2, game_id=gid)
        storage.save_df(dfs[0], path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_play_by_play.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add elevate_stat/fetchers/play_by_play.py tests/test_play_by_play.py
git commit -m "feat: per-game play-by-play fetcher"
```

---

## Task 7: Shots fetcher

**Files:**
- Create: `elevate_stat/fetchers/shots.py`
- Test: `tests/test_shots.py`

`ShotChartDetail` with `team_id=0, player_id=0` returns *all* shots league-wide for a season — so one call per season+type instead of one per player. One parquet per season+type.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_shots.py
import pandas as pd
from elevate_stat.fetchers import shots
from elevate_stat import storage


class FakeClient:
    def __init__(self):
        self.calls = []

    def call(self, factory, **kwargs):
        self.calls.append(kwargs)
        return [pd.DataFrame({"LOC_X": [1], "LOC_Y": [2], "SHOT_MADE_FLAG": [1]})]


def test_fetch_shots_uses_all_player_all_team_and_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    shots.fetch_shots("2015-16", client=client, season_types=["Regular Season"])
    assert storage.exists(storage.raw_path("shots", "2015-16_regular-season.parquet"))
    assert client.calls[0]["player_id"] == 0
    assert client.calls[0]["team_id"] == 0
    assert client.calls[0]["context_measure_simple"] == "FGA"


def test_fetch_shots_skips_existing(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    for _ in range(2):
        shots.fetch_shots("2015-16", client=client, season_types=["Regular Season"])
    assert len(client.calls) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_shots.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# elevate_stat/fetchers/shots.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_shots.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add elevate_stat/fetchers/shots.py tests/test_shots.py
git commit -m "feat: per-season all-player shot chart fetcher"
```

---

## Task 8: Aggregates fetcher (player-season stats, Synergy, lineups)

**Files:**
- Create: `elevate_stat/fetchers/aggregates.py`
- Test: `tests/test_aggregates.py`

Season-level endpoints (small). Player-season Base stats are fetched first because Task 9 (shot logs) needs the player-ID list, exposed here via `player_ids(season)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggregates.py
import pandas as pd
from elevate_stat.fetchers import aggregates
from elevate_stat import storage


class FakeClient:
    def __init__(self):
        self.calls = []

    def call(self, factory, **kwargs):
        self.calls.append((factory.__name__, kwargs))
        return [pd.DataFrame({"PLAYER_ID": [201939, 2544], "PTS": [30, 27]})]


def test_player_stats_writes_per_measure_and_exposes_player_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    aggregates.fetch_player_season("2015-16", client=client)
    assert storage.exists(storage.raw_path("player_season", "2015-16_base.parquet"))
    assert storage.exists(storage.raw_path("player_season", "2015-16_advanced.parquet"))
    assert sorted(aggregates.player_ids("2015-16")) == [2544, 201939]


def test_synergy_and_lineups_write_files(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    aggregates.fetch_lineups("2015-16", client=client, season_types=["Regular Season"])
    aggregates.fetch_synergy("2015-16", client=client, play_types=["Isolation"])
    assert storage.exists(storage.raw_path("lineups", "2015-16_regular-season.parquet"))
    assert storage.exists(storage.raw_path("synergy", "2015-16_isolation_offensive.parquet"))
    assert storage.exists(storage.raw_path("synergy", "2015-16_isolation_defensive.parquet"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_aggregates.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# elevate_stat/fetchers/aggregates.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_aggregates.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add elevate_stat/fetchers/aggregates.py tests/test_aggregates.py
git commit -m "feat: season-level aggregates (player stats, lineups, synergy)"
```

---

## Task 9: Shot-logs fetcher

**Files:**
- Create: `elevate_stat/fetchers/shot_logs.py`
- Test: `tests/test_shot_logs.py`

Per-player tracking shot logs (defender distance, touch time, dribbles). One parquet per player under `shot_logs/{season}/{player_id}.parquet`. This is the request-heavy endpoint (~500 players × 11 seasons), so skip-if-exists matters most here.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_shot_logs.py
import pandas as pd
from elevate_stat.fetchers import shot_logs
from elevate_stat import storage


class FakeClient:
    def __init__(self):
        self.calls = []

    def call(self, factory, **kwargs):
        self.calls.append(kwargs)
        return [pd.DataFrame({"CLOSE_DEF_DIST_RANGE": ["0-2 Feet"], "FGM": [1]})]


def test_fetch_writes_one_file_per_player(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    shot_logs.fetch_shot_logs("2015-16", [201939, 2544], client=client)
    assert storage.exists(storage.raw_path("shot_logs", "2015-16", "201939.parquet"))
    assert storage.exists(storage.raw_path("shot_logs", "2015-16", "2544.parquet"))
    assert len(client.calls) == 2


def test_fetch_skips_existing_players(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "RAW_DIR", tmp_path / "raw")
    client = FakeClient()
    shot_logs.fetch_shot_logs("2015-16", [201939], client=client)
    shot_logs.fetch_shot_logs("2015-16", [201939, 2544], client=client)
    assert len(client.calls) == 2  # 201939 once, 2544 once
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_shot_logs.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# elevate_stat/fetchers/shot_logs.py
from nba_api.stats.endpoints import PlayerDashPtShotLog
from elevate_stat import storage, client as _client


def fetch_shot_logs(season, player_ids, *, client=_client):
    for pid in player_ids:
        path = storage.raw_path("shot_logs", season, f"{pid}.parquet")
        if storage.exists(path):
            continue
        dfs = client.call(
            PlayerDashPtShotLog,
            player_id=pid,
            team_id=0,
            season=season,
            season_type_all_star="Regular Season",
        )
        storage.save_df(dfs[0], path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_shot_logs.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add elevate_stat/fetchers/shot_logs.py tests/test_shot_logs.py
git commit -m "feat: per-player tracking shot-log fetcher"
```

---

## Task 10: Orchestrator CLI

**Files:**
- Create: `elevate_stat/run_ingest.py`
- Test: `tests/test_run_ingest.py`

Walks seasons in dependency order (games → player-season → shots → shot-logs → play-by-play → synergy → lineups), logging progress. Pure wiring — it calls the real module-level `client.call` by default. CLI flags allow scoping a run to specific seasons for the live smoke test in Task 11.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_ingest.py
from elevate_stat import run_ingest


def test_parse_args_defaults_to_all_seasons():
    args = run_ingest.parse_args([])
    assert args.seasons is None


def test_parse_args_accepts_season_subset():
    args = run_ingest.parse_args(["--seasons", "2015-16", "2016-17"])
    assert args.seasons == ["2015-16", "2016-17"]


def test_ingest_season_calls_each_stage_in_order(monkeypatch):
    order = []
    monkeypatch.setattr(run_ingest.games, "fetch_games", lambda s: order.append("games"))
    monkeypatch.setattr(run_ingest.games, "game_ids", lambda s: ["001"])
    monkeypatch.setattr(run_ingest.aggregates, "fetch_player_season", lambda s: order.append("player_season"))
    monkeypatch.setattr(run_ingest.aggregates, "player_ids", lambda s: [1])
    monkeypatch.setattr(run_ingest.aggregates, "fetch_synergy", lambda s: order.append("synergy"))
    monkeypatch.setattr(run_ingest.aggregates, "fetch_lineups", lambda s: order.append("lineups"))
    monkeypatch.setattr(run_ingest.shots, "fetch_shots", lambda s: order.append("shots"))
    monkeypatch.setattr(run_ingest.shot_logs, "fetch_shot_logs", lambda s, pids: order.append("shot_logs"))
    monkeypatch.setattr(run_ingest.play_by_play, "fetch_play_by_play", lambda s, gids: order.append("pbp"))

    run_ingest.ingest_season("2015-16")

    assert order == ["games", "player_season", "shots", "shot_logs", "pbp", "synergy", "lineups"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_run_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# elevate_stat/run_ingest.py
import argparse
import logging
import sys
from elevate_stat import config
from elevate_stat.fetchers import games, play_by_play, shots, aggregates, shot_logs

log = logging.getLogger("elevate_stat.ingest")


def ingest_season(season: str) -> None:
    log.info("=== %s: games ===", season)
    games.fetch_games(season)
    gids = games.game_ids(season)

    log.info("=== %s: player-season stats ===", season)
    aggregates.fetch_player_season(season)
    pids = aggregates.player_ids(season)

    log.info("=== %s: shots ===", season)
    shots.fetch_shots(season)

    log.info("=== %s: shot logs (%d players) ===", season, len(pids))
    shot_logs.fetch_shot_logs(season, pids)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_run_ingest.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/pytest`
Expected: PASS (all tests, ~20)

- [ ] **Step 6: Commit**

```bash
git add elevate_stat/run_ingest.py tests/test_run_ingest.py
git commit -m "feat: ingest orchestrator CLI"
```

---

## Task 11: Live smoke test, then launch the full scrape

**Files:** none (operational task)

- [ ] **Step 1: Live smoke test — one season, connectivity check**

Run:
```bash
.venv/bin/python -m elevate_stat.run_ingest --seasons 2015-16 2>&1 | tee ingest-smoke.log
```
Let it run for ~2–3 minutes, then confirm real files are landing:
```bash
find data/raw -name '*.parquet' | head
.venv/bin/python -c "import pandas as pd, glob; f=glob.glob('data/raw/games/*.parquet')[0]; print(pd.read_parquet(f).shape)"
```
Expected: parquet files appear under `data/raw/games/`, `data/raw/player_season/`, etc., and the printed shape is non-empty. If calls hang/timeout repeatedly, stop and diagnose headers/rate-limit before the full run (do NOT launch the multi-hour job on a broken connection).

- [ ] **Step 2: Sanity-check the smoke data, then let 2015-16 finish**

Once connectivity is confirmed, allow the `--seasons 2015-16` run to complete (this alone is a meaningful chunk — one season of play-by-play is the bulk of a season's requests). Confirm it finishes without unhandled exceptions.

- [ ] **Step 3: Launch the full background scrape**

Run (background, survives the session, logs to file):
```bash
nohup .venv/bin/python -m elevate_stat.run_ingest > ingest-full.log 2>&1 &
echo "ingest PID: $!"
```
This walks all 11 seasons. Because every fetcher skips already-downloaded files, the already-complete 2015-16 data is not re-pulled. Monitor with:
```bash
tail -f ingest-full.log
find data/raw -name '*.parquet' | wc -l   # file count grows over time
```

- [ ] **Step 4: Add `ingest-*.log` to `.gitignore` and commit**

```bash
printf '\n# ingest run logs\ningest-*.log\n' >> .gitignore
git add .gitignore
git commit -m "chore: ignore ingest run logs"
```

- [ ] **Step 5: Push all work**

```bash
git push
```
Expected: all Phase 0 commits land on `origin/main`. The scrape keeps running in the background locally; no data is pushed (it's gitignored).

---

## Self-Review Notes

- **Spec coverage (§6 manifest):** play-by-play ✓ (Task 6), shots/location ✓ (Task 7), tracking shot logs ✓ (Task 9), lineups ✓ (Task 8), Synergy ✓ (Task 8), player-season stats ✓ (Task 8). Box scores are deferred — on/off is reconstructed from play-by-play (§3), so `BoxScoreAdvancedV2` is not needed for Phase 0 and is intentionally omitted (YAGNI).
- **Spec coverage (§7 pipeline):** resumable ✓ (file-existence skip, every fetcher), rate-limited + retry/backoff ✓ (Task 4), checkpointed to parquet ✓ (Task 3), background job ✓ (Task 11).
- **Type consistency:** every fetcher takes `client=` and calls `client.call(EndpointClass, **kwargs)`; `FakeClient.call(self, factory, **kwargs)` matches; `game_ids`/`player_ids` are read back by the orchestrator with the same signatures they're defined with.
- **Known live-run risks (not blockers, watch during Task 11):** `nba_api` header staleness can cause persistent timeouts — if so, the retry wrapper surfaces a clear `RuntimeError`; the smoke test (Task 11 Step 1) exists to catch this before the long run. `PlayerDashPtShotLog` for players with zero tracked shots may return an empty frame — that's fine, it writes an empty parquet and is skipped on resume.
