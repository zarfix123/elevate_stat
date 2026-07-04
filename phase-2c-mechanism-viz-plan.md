# Phase 2c — Mechanism Lens + Visualizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Decompose *why* a player elevates teammates (creating shots vs. improving shot quality), stabilize the metric with shrinkage, and render the final visualizations (clutch scatter, mechanism map, career trajectory).

**Architecture:** Extend `models/elevation_teammate.py` with `compute_mechanism`, which splits each A→B lift into a **volume channel** (B takes more shots with A on = creation/passing) and an **efficiency channel** (B's shots are better with A on = spacing/gravity), with sample-size shrinkage. Add three chart functions to `viz.py`. `build_elevation.py` renders them on the full run.

**Tech Stack:** Python, pandas/numpy, matplotlib, pytest.

**Methodology (locked):** For teammate B, `scoring_rate = pts/poss = (pts/tsa) × (tsa/poss) = efficiency × volume`. So the two channels (Δefficiency, Δvolume) *are* the mechanism of B's scoring lift. Defense channel needs defensive tracking we don't have → out of scope (stated). Each pair lift is shrunk by `shared_tsa / (shared_tsa + K)` toward 0, so thin pairs don't dominate — the honest fix for the residual role-player confound (full ridge-adjustment is left as future research).

---

## File Structure
```
elevate_stat/
  models/elevation_teammate.py   # + compute_mechanism()
  viz.py                          # + clutch_scatter, mechanism_map, trajectory
  build_elevation.py             # + render the 3 new charts
tests/
  test_elevation_teammate.py     # + mechanism tests
  test_viz.py                    # + 3 new chart tests
```
**Artifacts:** `data/processed/mechanism.parquet` (per player: vol_centrality, eff_centrality, minutes); `figures/{clutch_scatter,mechanism_map,trajectory}.png`.

---

## Task 1: Mechanism decomposition (`models/elevation_teammate.py`)
Add `compute_mechanism(stints_df, min_shared_tsa=150.0, shrink_k=400.0)`. Accumulate per player and per pair: leverage-weighted `pts`, `tsa`, `poss` (poss = seconds/28.8). For each qualifying pair (A,B):
- `eff_lift = 100 * (pts_with/tsa_with − pts_without/tsa_without)`
- `vol_lift = 100 * (tsa_with/poss_with − tsa_without/poss_without)`
- `shrink = shared_tsa / (shared_tsa + shrink_k)`; multiply both lifts by `shrink`.
Per player A: `eff_centrality`, `vol_centrality` = shared-tsa-weighted means. Returns `(mech_df[PLAYER_ID, vol_centrality, eff_centrality, minutes], pairs_df[A, B, eff_lift, vol_lift, shared_tsa])`.

TDD (`tests/test_elevation_teammate.py`): synthetic stints where player 1's teammate takes MORE shots with 1 on (same efficiency) -> 1 has positive `vol_centrality`, ~0 `eff_centrality`; a second dataset where teammate is more efficient but same volume -> positive `eff_centrality`, ~0 `vol_centrality`. Commit.

## Task 2: New visualizations (`viz.py`)
Three pure functions (Agg backend), each writes a PNG:
- `clutch_scatter(ratings_df, path, names, min_minutes=12000)` — scatter `rapm` (x) vs `late` (y) from `late_ratings.parquet`; y=x reference line; annotate the biggest positive `(late-rapm)` risers. Title "Who rises when it matters (above line = clutch)".
- `mechanism_map(mech_df, path, names, n=18)` — scatter `vol_centrality` (x, "creates shots") vs `eff_centrality` (y, "improves shot quality") for the top-elevator set; annotate points. Quadrant lines at 0.
- `trajectory(metrics_df, player_ids, names, path)` — line chart of per-season `points_above_expected` (shot-making, from Phase-1 `xpps_player_metrics.parquet`) for a few players across seasons.

TDD (`tests/test_viz.py`): each writes a non-empty PNG on tiny synthetic input. Commit.

## Task 3: Build integration + real run + face validity (`build_elevation.py`)
Add to `main`: compute mechanism, write `mechanism.parquet`, and render the 3 new charts (load `late_ratings.parquet` for the clutch scatter and `xpps_player_metrics.parquet` for the trajectory). Guard each render behind an existence check on its input.

**Operational (real data):** run `python -m elevate_stat.build_elevation`. Verify **face validity of the mechanism split**: pass-first playmakers (Chris Paul, LeBron, Haliburton, Trae Young) should skew **volume** (create shots); shooters/gravity (Curry, Klay-lifting bigs) should skew **efficiency**. Render + **send the user** the clutch scatter, mechanism map, and a trajectory (e.g., Jokić / Curry / a rising young star). Fix issues, commit.

---

## Self-Review Notes
- **Spec coverage:** design §5.2 mechanism lens -> Tasks 1+3 (volume+efficiency channels; defense channel explicitly out of scope for lack of defensive tracking); more viz §10 -> Task 2. Ridge-adjusted residual is explicitly deferred (stated as research); shrinkage (Task 1) is the pragmatic stabilization.
- **Types:** `compute_mechanism` returns `(mech_df, pairs_df)` with the columns above; viz functions take DataFrames + path and return the path. Consistent with Phase-2b interfaces.
- **Depends on:** `data/processed/late_ratings.parquet` (Phase 2a) and `xpps_player_metrics.parquet` (Phase 1) for two of the charts — renders are guarded on their presence.
