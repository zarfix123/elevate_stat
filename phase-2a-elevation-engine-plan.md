# Phase 2a — Elevation Engine Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Turn play-by-play into a stint dataset (who was on the floor, points, time, leverage) and fit a **leverage-weighted regularized impact metric** — the rigorous foundation of LATE — verified by face validity.

**Architecture:** `pbp_lineups.py` reconstructs the on-court 5-per-team through each game from substitution events + inferred period starters. `stints.py` turns a game into stint rows (constant 10-man segments: the two lineups, point margin, seconds, per-player points, and a leverage weight from the Phase-1 win-prob model). `models/elevation.py` stacks all stints into a sparse design matrix and fits ridge regression → per-player points-per-100 impact (offense/defense/total), leverage-weighted. A `build_late.py` CLI runs it across the dataset and writes ratings.

**Tech Stack:** Python, pandas/numpy, scipy.sparse, scikit-learn (Ridge), joblib, pytest.

**Methodology decisions (locked, with rationale):**
- **Stint-based, not per-possession.** A stint is a maximal segment with a constant 10-man lineup (breaks on any substitution or period boundary). Avoids fragile possession parsing; standard RAPM prep.
- **Time-weighted.** Weight each stint by `seconds × leverage`, and scale margins to per-100-possessions using a global seconds→possession constant (`SECONDS_PER_POSS ≈ 13.5` shared, i.e. ~2 possessions / 27s). Robust, sidesteps possession-counting edge cases.
- **Leverage integration.** Leverage for a stint = `win_prob.leverage(model, score_diff, seconds_remaining)` at the stint midpoint. This is what makes it LATE, not vanilla RAPM.
- **Core v1 metric = leverage-weighted RAPM** (offense/defense/total per 100). This is the rigorous, buildable foundation; the *counterfactual teammate-elevation* refinement and the 3 lenses (Phase 2b) build on this exact stint dataset. A raw leverage-weighted **teammate on/off** number is also emitted as an early elevation view (honestly labeled as unadjusted).
- **Pooled across the tracking era** for a stable single rating per player (single-season RAPM is too noisy). Per-season is a later extension.

---

## File Structure
```
elevate_stat/
  pbp_lineups.py          # reconstruct on-court lineups per event; validate 5v5
  stints.py               # game pbp -> stint rows (lineups, margin, secs, leverage, per-player pts)
  models/elevation.py     # sparse design + ridge -> per-player impact; teammate on/off
  build_late.py           # CLI: all games -> stint table -> fit -> data/processed/late_ratings.parquet
tests/
  test_pbp_lineups.py
  test_stints.py
  test_elevation.py
  test_build_late.py
```
**Artifacts (`data/processed/`, gitignored):** `late_ratings.parquet` (per player: o_rating, d_rating, total_rating, minutes, on_off_teammate), `elevation_meta.json` (stint count, 5v5 resolution rate, alpha).

---

## Task 1: Lineup reconstruction (`pbp_lineups.py`)

Parse one game's pbp into on-court sets. Substitution rows: `personId` = player OUT; incoming name parsed from `description` (`"SUB: <IN> FOR <OUT>"`) and resolved to an id via the game's name→id map. Period starters are **inferred**: processing events in order, the first time a player appears (any personId event, or is subbed out) before being subbed in, they were on the floor at period start.

Key functions:
- `name_to_id(game_pbp)` → dict of last-name → personId per team (from non-sub events).
- `parse_sub(description)` → incoming last-name.
- `reconstruct(game_pbp)` → adds `home_on` / `away_on` columns (frozensets of 5 personIds) to each event row; raises/flags if any period can't resolve to exactly 5 per team.
- `resolution_ok(game_pbp)` → bool (every stint has exactly 5 per team).

TDD:
- [ ] `parse_sub("SUB: Niang FOR Allen") == "Niang"`.
- [ ] On a hand-built 1-period synthetic game (5 starters/team who each record an event, then one sub), `reconstruct` yields exactly 5 per team throughout, and the sub swaps correctly.
- [ ] A player subbed out before recording any event is still counted as a starter (discovered via the sub).
- [ ] `resolution_ok` is False for a deliberately broken game (only 4 discoverable starters) — so downstream can skip it.
Commit.

## Task 2: Stint dataset (`stints.py`)

`home_away_from_games(season)` → dict `gameId -> (home_team_id, away_team_id)` from the games index (`MATCHUP` contains `"vs."` for home). `build_stints(game_pbp, home_id, away_id, wp_model)` → DataFrame with one row per stint:
- `home_lineup` (frozenset of 5), `away_lineup` (5)
- `home_pts`, `away_pts` (forward-filled `scoreHome/scoreAway` deltas across the stint)
- `seconds` (clock delta within period; stints break at period boundaries)
- `leverage` (`win_prob.leverage` at stint-midpoint score_diff + seconds_remaining)
- per-player points dict for the stint (from Made Shot / Free Throw rows with personId → `shotValue`/pts)

TDD (on a synthetic reconstructed game):
- [ ] a stint's `home_pts`/`away_pts` equal the score deltas over that segment;
- [ ] a substitution starts a new stint (lineup change → new row);
- [ ] `seconds` is positive and a blowout-late stint gets a lower `leverage` than a tie-late one (using a fitted toy wp model);
- [ ] `home_away_from_games` maps a `"CLE vs. IND"` row to CLE home.
Commit.

## Task 3: Elevation model (`models/elevation.py`)

`fit(stints_df, alpha=2000)`:
- Build player index (all ids appearing). Design matrix `X` (scipy CSR): one row per stint, `+1` for home lineup players, `−1` for away; response `y` = `(home_pts − away_pts) / possessions` scaled to per-100 (`possessions = seconds / SECONDS_PER_POSS`); sample weight `w = possessions × leverage`.
- `Ridge(alpha=alpha, fit_intercept=True)` on `(X, y, sample_weight=w)`. Coefficients = per-player **total** leverage-weighted impact per 100.
- Offense/defense split: fit two more ridge models where `y` is the offensive team's points-per-100 for/against, with lineup coded by which side had the ball (derive off/def stints by duplicating each stint into an offensive row for the scoring team). v1: total rating from the margin model; o/d from an offense-possession stacking. Keep it as `total_rating` (margin) plus `o_rating`/`d_rating` from the stacked model.
- `teammate_on_off(stints_df)`: for each player, leverage-weighted offensive points-per-100 of their team when on court minus league baseline, EXCLUDING their own points (uses per-player points) — the unadjusted elevation view.
- Returns a per-player DataFrame: `PLAYER_ID, total_rating, o_rating, d_rating, minutes, on_off_teammate`.

TDD: on a small synthetic stint set where one player is present in every high-margin stint, that player gets the highest `total_rating`; ridge shrinks a tiny-sample player toward 0; `teammate_on_off` excludes the player's own points.
Commit.

## Task 4: build_late CLI + real run + face validity (`build_late.py`)

CLI loads the Phase-1 `win_prob_model.joblib`, iterates all seasons/games, reconstructs + builds stints (skipping games where `resolution_ok` is False, logging the skip rate), concatenates, fits `elevation.fit`, joins player names, writes `late_ratings.parquet` + meta. Test with monkeypatched loaders on a tiny synthetic game asserting artifacts written.

**Operational (real data):** run `python -m elevate_stat.build_late`. Verify:
- 5v5 resolution rate > ~90% (log it; investigate if lower).
- Face validity: the top of the `total_rating` leaderboard (min ~3000 minutes) is populated by stars/known high-impact players (Jokić, Curry, LeBron, Draymond-type), not random low-minute names; ridge alpha tuned so the leaderboard is stable (not dominated by tiny samples).
- Sanity: ratings roughly center near 0, spread of a few points per 100.
Fix issues, commit.

---

## Self-Review Notes
- **Spec coverage:** design §3 core engine (leverage-weighted, regularized, on lineups) → Tasks 1–4. Counterfactual **teammate** residual (full) and the **3 lenses** (§5) + viz (§10) are explicitly Phase 2b — this plan delivers the impact core + an unadjusted teammate on/off as the first elevation view.
- **Data reality baked in:** substitution `personId`=out + description parse (verified in pbp); `scoreHome/away` forward-fill (verified sparse); inferred period starters (no starter list in pbp); home/away from games `MATCHUP`.
- **Highest risk = lineup reconstruction** → Task 1 is validation-heavy and Task 4 skips + logs unresolvable games rather than corrupting the model.
- **Types:** `reconstruct` returns the pbp frame + `home_on`/`away_on` frozenset columns; `build_stints` returns a stint DataFrame; `elevation.fit` returns a per-player DataFrame keyed by `PLAYER_ID`. Consistent across tasks.
- **Depends on Phase 1 artifact** `data/processed/win_prob_model.joblib` (leverage). Build_late errors clearly if it's missing.
