# Phase 2b — Teammate Elevation, Network & Archetype Lenses, Visualizations

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Turn the stint data into a **teammate-elevation** metric (does a player make teammates *score* more?), expose it through the **network** (who-lifts-whom) and **archetype** lenses, and render the visualizations.

**Architecture:** Add per-player points to each stint (from the pbp Made-Shot/Free-Throw rows). `models/elevation_teammate.py` computes leverage-weighted WOWY: for every player, teammate scoring-rate *with them on* vs *off*, aggregated into an **elevation-centrality** and a directed **A→B lift** matrix; the archetype lens conditions lift on the teammate's Phase-1 playstyle archetype. `viz.py` renders leaderboard, who-lifts-whom network, and archetype-lift charts to PNG. A `build_elevation.py` CLI runs it on the full dataset and writes ratings + figures.

**Tech Stack:** Python, pandas/numpy, matplotlib, networkx, pytest. (New deps: matplotlib, networkx.)

**Methodology (locked, honest):** v1 elevation is **leverage-weighted WOWY on teammate scoring** — interpretable and directional (needed for the network), but *unadjusted* (on/off confounding remains; ridge-adjustment is a later refinement). Response = a teammate's points per 100 possessions; leverage weights emphasize high-stakes possessions. Minutes/shared-possession thresholds + shrinkage keep it stable.

---

## File Structure
```
elevate_stat/
  stints.py                     # + per-player points per stint (augment)
  models/elevation_teammate.py  # WOWY teammate lift, elevation-centrality, A->B matrix, archetype lens
  viz.py                        # matplotlib/networkx charts -> PNG
  build_elevation.py            # CLI: full run -> data/processed/elevation_teammate.parquet + figures/
tests/
  test_elevation_teammate.py
  test_viz.py
```
**Artifacts:** `data/processed/elevation_teammate.parquet` (per player: elevation_centrality, minutes, name); `data/processed/pairs.parquet` (A,B,lift,shared_poss); `data/processed/figures/*.png`.

---

## Task 1: Per-player points per stint (augment `stints.py`)
Add a `player_pts` column (dict `personId -> points`) to `build_stints`, summed over the stint's own events: `Made Shot` rows add `shotValue` to `personId`; `Free Throw` rows whose description lacks "MISS" add 1.

TDD (extend `tests/test_stints.py`): on the synthetic game, stint 0's `player_pts` gives P1=2 and P2=2 (their made shots), and stint 1 gives P6=2. Commit.

## Task 2: Teammate-elevation WOWY (`models/elevation_teammate.py`)
`compute(stints_df, seconds_per_poss=28.8, min_shared_poss=100)`:
- Accumulate leverage-weighted `pts` and `poss` per player (`player_tot`) and per ordered pair `(A,B)` for teammates sharing a lineup (`pair` -> B's pts & poss while A on). Weight = leverage: accumulate `leverage*B_pts` and `leverage*poss`.
- `lift(A->B) = 100 * (B_rate_with_A - B_rate_without_A)` where without = `(tot - with) / (poss_tot - poss_with)`, only when both `poss_with` and `poss_without` exceed `min_shared_poss`.
- `elevation_centrality(A) = shared-poss-weighted mean of lift(A->B) over B`.
- Returns `(centrality_df[PLAYER_ID, elevation_centrality, minutes], pairs_df[A, B, lift, shared_poss])`.

TDD: synthetic stints where teammates of player 1 score at a high rate only when 1 is on -> player 1 has positive centrality and lift(1->teammate) > 0; a player with no differential has ~0; `min_shared_poss` filters thin pairs. Commit.

## Task 3: Archetype lens (`models/elevation_teammate.py`)
`elevation_by_archetype(pairs_df, archetypes)` where `archetypes` maps `PLAYER_ID -> top_archetype` (from Phase-1 `data/processed/playstyle/*`): for each elevator A, the shared-poss-weighted mean lift onto teammates *of each archetype* -> long DataFrame `[A, archetype, lift, shared_poss]`.

TDD: synthetic pairs where A lifts archetype-0 teammates strongly and archetype-1 weakly -> the returned per-archetype means reflect that ordering. Commit.

## Task 4: Visualizations (`viz.py`)
Pure functions that take data + an output path and write a PNG (headless `matplotlib.use("Agg")`):
- `centrality_bar(centrality_df, path, n=20)` — horizontal bar of top elevators.
- `who_lifts_whom(pairs_df, players, names, path)` — directed `networkx` graph among a chosen player set; edge width/color by lift.
- `archetype_bar(arch_df, player_id, name, path)` — a player's lift by teammate archetype.

TDD (`tests/test_viz.py`): each function writes a non-empty `.png` to a tmp path on tiny synthetic input (assert file exists and size > 0). Commit.

## Task 5: build CLI + real run + face validity (`build_elevation.py`)
CLI reuses `build_late.build_stint_table` (now carrying `player_pts`), runs `compute`, loads Phase-1 playstyle archetypes for the archetype lens, writes ratings/pairs parquet, and renders the figures. Test with a mocked tiny stint table asserting the parquet + at least one PNG are written.

**Operational (real data):** run `python -m elevate_stat.build_elevation`. Verify **face validity**: the top of `elevation_centrality` (min ~5000 min) should be **playmakers / gravity players** (Jokić, LeBron, Chris Paul, Curry, Draymond, Harden, Haliburton) — not random names. Render and **send the user** the centrality leaderboard, a who-lifts-whom network for a marquee team (e.g. 2016-18 Warriors or a Jokić Nuggets roster), and an archetype-lift chart. Fix issues, commit.

---

## Self-Review Notes
- **Spec coverage:** design §5.1 network lens -> Tasks 2+4; §5.3 archetype lens -> Tasks 3+4; teammate-elevation core -> Tasks 1-2; visualizations §10 -> Task 4. §5.2 **mechanism lens** is explicitly deferred to Phase 2c (hardest / most speculative).
- **Honest limitation:** WOWY is unadjusted (confounded); face-validity + thresholds are the guardrails; ridge-adjusted teammate residual is future work. Stated in output.
- **Types:** `build_stints` gains a `player_pts` dict column; `compute` returns `(centrality_df, pairs_df)`; viz functions take DataFrames + path, return the path. Consistent.
- **Deps:** matplotlib + networkx added to requirements; Agg backend for headless PNG.
