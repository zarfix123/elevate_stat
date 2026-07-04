# Phase 1 — Shared Engines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the three shared modeling engines the LATE metric depends on — win-probability/leverage, expected-points-per-shot (xPPS), and playstyle-DNA clusters — each fit on the local dataset and producing saved artifacts.

**Architecture:** A `elevate_stat/data.py` loader layer reads the raw parquet (selective columns, per season) so the models never touch file paths directly. Three focused modules under `elevate_stat/models/` each expose a `fit(...)` + an apply function and are independently testable on tiny synthetic frames. A `build_models.py` CLI fits all three on the full dataset and writes artifacts to `data/processed/` (gitignored). ML is built iteratively with real-data smoke runs, not just unit tests, because the models must produce *sane* output, not merely run.

**Tech Stack:** Python, pandas/numpy, scikit-learn (LogisticRegression, HistGradientBoostingClassifier, GaussianMixture, StandardScaler), joblib (model persistence), pytest.

**Design note (leverage as a reusable lens):** the win-prob module is the foundation for the eventual leverage-weighted elevation metric — it must expose per-state win probability AND a leverage index, not just a classifier.

---

## File Structure

```
elevate_stat/
  data.py                 # raw-parquet loaders (per season, selective columns)
  models/
    __init__.py
    win_prob.py           # clock parsing, WP logistic model, leverage index
    xpps.py               # expected points per shot + player shot metrics
    playstyle.py          # player style-feature matrix, GMM archetypes
  build_models.py         # CLI: fit all three on full dataset -> data/processed/
tests/
  test_data.py
  test_win_prob.py
  test_xpps.py
  test_playstyle.py
  test_build_models.py
```

**Artifacts written to `data/processed/` (gitignored):**
- `win_prob_model.joblib`, `win_prob_meta.json` (feature spec + a leverage-normalization constant)
- `xpps_model.joblib`; `xpps_player_metrics.parquet` (per player-season: points_above_expected, avg_xpps)
- `playstyle_model.joblib` (scaler+gmm+feature list); `playstyle/{season}_{type}.parquet` (per player-season archetype memberships)

---

## Design decisions (locked)

**Win-probability / leverage (`win_prob.py`)**
- One training row per pbp event: `score_diff` (home − away, from running `scoreHome/scoreAway`), `seconds_remaining` (game-level: `(4-period)*720 + clock_secs` for period≤4; `clock_secs` for OT), label `home_won` (from each game's final score).
- Model: `LogisticRegression` on features `[score_diff, sqrt_time, score_diff/sqrt(seconds_remaining+1)]` where `sqrt_time = sqrt(seconds_remaining)`. This captures "a lead matters more as time shrinks."
- `win_probability(model, score_diff, seconds_remaining) -> float`.
- `leverage(model, score_diff, seconds_remaining)` = `|WP(score_diff+2) − WP(score_diff−2)|` at that time — the win-prob swing across a ~one-possession margin change. High late-and-close, ~0 in blowouts. Store the dataset-mean swing in meta so a normalized leverage (mean≈1.0) is available later.

**Expected points per shot (`xpps.py`)**
- Features from the shots table: `SHOT_DISTANCE`, `LOC_X`, `LOC_Y`, one-hot `SHOT_ZONE_BASIC`, `is_three` (from `SHOT_TYPE` contains "3PT"). Label `SHOT_MADE_FLAG`.
- Model: `HistGradientBoostingClassifier` (fast on ~2.5M shots) predicting P(make).
- `expected_points(model, shots_df)` = `P(make) * point_value` where point_value = 3 if is_three else 2.
- `player_shot_metrics(shots_df, xpps_series)` → per (PLAYER_ID, season): `shots`, `actual_pts`, `expected_pts`, `points_above_expected` (shot-MAKING skill), `avg_xpps` (shot-SELECTION).

**Playstyle-DNA (`playstyle.py`)**
- Per player-season feature vector, min ~500 MIN filter: from advanced (`USG_PCT, AST_PCT, OREB_PCT, DREB_PCT, TS_PCT, PACE`), from base rates (`FG3A/FGA`, `FTA/FGA`, `AST/ (FGA+0.44*FTA+TOV)` assist load, `BLK+STL` per-min defensive events), plus synergy playtype `POSS_PCT` for each of the 11 play types (offensive grouping), plus shot-zone distribution (share of shots per `SHOT_ZONE_BASIC`).
- Model: `StandardScaler` → `GaussianMixture(n_components=8, covariance_type='full', random_state=0)`. Soft membership per player.
- `assign(...)` → per player-season: membership prob per cluster + `top_archetype`.

---

## Task 1: Data loader layer

**Files:** Create `elevate_stat/data.py`; Test `tests/test_data.py`

- [ ] **Step 1 — failing test** (`tests/test_data.py`): with `RAW_DIR` monkeypatched to a tmp dir holding a couple of parquet files, `data.load_shots("2015-16", "Regular Season")` returns the concatenated frame; `data.pbp_game_paths("2015-16")` lists game files.
- [ ] **Step 2** run → fails (no module).
- [ ] **Step 3 — implement** `elevate_stat/data.py`:
```python
import glob
from pathlib import Path
import pandas as pd
from elevate_stat import config, storage

def _slug(s): return s.lower().replace(" ", "-")

def pbp_game_paths(season):
    return sorted(glob.glob(str(storage.raw_path("play_by_play", season, "*.parquet"))))

def load_shots(season, season_type, columns=None):
    p = storage.raw_path("shots", f"{season}_{_slug(season_type)}.parquet")
    return pd.read_parquet(p, columns=columns) if p.exists() else pd.DataFrame()

def load_player_season(season, season_type, measure, columns=None):
    p = storage.raw_path("player_season", f"{season}_{_slug(season_type)}_{measure.lower()}.parquet")
    return pd.read_parquet(p, columns=columns) if p.exists() else pd.DataFrame()

def load_synergy(season, season_type, grouping="offensive"):
    paths = glob.glob(str(storage.raw_path("synergy", f"{season}_{_slug(season_type)}_*_{grouping}.parquet")))
    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
```
- [ ] **Step 4** run → passes. **Step 5** commit.

## Task 2: Win-probability + leverage engine
TDD, in order: (a) `parse_clock("PT10M25.00S")==625.0`; (b) `seconds_remaining(period, secs)` monotonic decreasing across the game; (c) `build_training_frame(game_pbp)` yields score_diff/seconds_remaining/home_won with home_won constant per game = (final home>away); (d) after `fit` on a synthetic separable frame, `win_probability` ↑ with score_diff and →~0.5 at tie with large time; (e) `leverage` larger for (diff=0, secs=30) than (diff=0, secs=2000) and larger than (diff=25, secs=30). Then commit.

## Task 3: xPPS engine
TDD: (a) `build_features` returns numeric X with an `is_three` column and y=made flag; (b) after `fit` on synthetic shots (short makes, long misses), predicted make-prob is higher for a short shot; (c) `expected_points` multiplies by 3 for a three; (d) `player_shot_metrics` aggregates points_above_expected and avg_xpps correctly on a tiny frame. Commit.

## Task 4: Playstyle-DNA engine
TDD: (a) `build_player_features` merges advanced+synergy+shot-dist into one row-per-player numeric frame with no NaNs after fill; (b) after `fit(n_components=3)` on three well-separated synthetic profile clusters, `assign` puts each profile in a distinct top_archetype and membership rows sum to ~1.0. Commit.

## Task 5: build_models CLI
`elevate_stat/build_models.py` with `main()` that: (1) builds WP training frame by sampling ~3000 games across seasons (load pbp selective columns), fits, saves model+meta; (2) loads all shots (both season types, all seasons), fits xPPS, writes player metrics parquet + model; (3) builds playstyle features for every player-season (RS), fits GMM, writes memberships + model. Test with monkeypatched loaders returning tiny frames + `data/processed` in tmp, asserting all artifacts are written. Commit.

## Task 6: Real-data build + sanity
Operational: run `python -m elevate_stat.build_models` on the full dataset. Then sanity-check the outputs:
- WP: `win_probability(diff=0, secs=1440)≈0.5`; a +10 lead with 2 min left → >0.9; leverage of a tie-game final possession ≫ leverage of a 20-pt blowout.
- xPPS: restricted-area shots have high xPPS (~1.2+), long twos low; the points_above_expected leaderboard's top names are elite shooters/finishers (face validity).
- Playstyle: eyeball a few archetype clusters (e.g., high-USG_PCT/AST_PCT guards vs. low-USG rim-running bigs) for coherence.
Fix any issues, commit.

---

## Self-Review Notes
- **Spec coverage:** design §4.1 win-prob/leverage → Task 2; §4.2 xPPS baseline → Task 3; §4.3 playstyle-DNA → Task 4; all wired + run in Tasks 5–6. Data loaders (Task 1) are shared infra.
- **Out of scope (correctly deferred to Phase 2):** the elevation engine itself, lineups/on-off (`data/raw/lineups` untouched here), the 3 lenses, visualizations.
- **Types:** loaders return `pd.DataFrame`; each engine's `fit` returns an sklearn estimator (persisted via joblib); apply functions take `(model, df)` and return a `Series`/`DataFrame`. Consistent across tasks.
- **Known risk:** loading all pbp is heavy → Task 5 *samples* games for WP training (logistic needs far less than 7M rows) and logs the sample size; xPPS uses all shots (~2.5M rows, fine for HistGBM).
