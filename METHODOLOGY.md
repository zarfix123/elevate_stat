# LATE — Leverage-Adjusted Teammate Elevation

**A from-scratch NBA impact + teammate-elevation metric, built entirely from public play-by-play.**

This is the full technical documentation: what the metric is, the science and math behind every component, the code that implements it, and — importantly — the honest limitations. Everything here is reproducible from the code in this repo.

- **Data:** 2015-16 → 2025-26 (the NBA "tracking era"), regular season **and** playoffs — 14,128 games.
- **Repo:** `github.com/zarfix123/elevate_stat`
- **Interactive explorer:** the static site under `docs/` (player search, who-affects-who web, reg/playoff/both, season range).
- **Design spec (concept):** `teammate-elevation-design.md`. **Per-phase build plans:** `phase-0…2c-*.md`.

---

## 1. Motivation & lineage

Modern "all-in-one" NBA impact metrics — RAPTOR (FiveThirtyEight), LEBRON (BBall-Index), EPM, DARKO — all output the same thing: **points contributed per 100 possessions, relative to a league-average player**, blending a box-score prior with an on/off (plus-minus) signal. LATE sits in that family but makes two deliberate, less-explored choices:

1. **Leverage-weighting.** Every possession is weighted by how much it can swing win probability (borrowed from baseball's Leverage Index). A tie game with two minutes left counts heavily; a 25-point blowout counts ~0. This is standard in baseball (WPA/LI) but essentially absent from public basketball impact metrics.
2. **Teammate elevation.** Beyond a player's own impact, LATE measures whether a player makes *teammates* more efficient — and decomposes *how* (creating shots vs. improving shot quality) and *for whom* (by teammate playstyle archetype).

Everything is built from public data only (`nba_api` / stats.nba.com), reconstructing on-court lineups from raw play-by-play rather than using any private tracking feed.

---

## 2. Data layer

### 2.1 Sources (`nba_api`)
| Dataset | Endpoint | Used for |
|---|---|---|
| Play-by-play | `PlayByPlayV3` | lineup reconstruction, stints, scoring, leverage |
| Shot locations | `ShotChartDetail` | expected-points (xPPS) model |
| Tracking shots | `LeagueDashPlayerPtShot` (defender-distance buckets) | xPPS context |
| Playtypes | `SynergyPlayTypes` | playstyle-DNA features |
| Player-season stats | `LeagueDashPlayerStats` (Base/Advanced/Scoring/Usage) | playstyle-DNA, names, teams |
| Lineups | `LeagueDashLineups` | reference/validation |

**Scope choice (2015-16 start):** Synergy playtypes begin in 2015-16, so the window starts there to keep every season's feature set identical.

### 2.2 Ingestion pipeline (`elevate_stat/`)
- `config.py` — season list + tuning constants. `storage.py` — parquet I/O with **atomic writes** (temp file + `os.replace`) so a crash never leaves a half-written file that resume logic would skip. `client.py` — a rate-limited, retrying wrapper around `nba_api`. `fetchers/*` — one module per endpoint family, all following "skip if the output file exists, else fetch + save" (so any interrupted run resumes for free). `run_ingest.py` — the orchestrator.
- `scripts/supervise_ingest.sh` — a self-healing supervisor that re-runs the resumable ingest until a full pass adds nothing new, repairing partial files between passes. This survived a live stats.nba.com throttle mid-run (27 games deferred, then auto-backfilled).
- Raw data lands in `data/raw/` (gitignored — ~550 MB of parquet, never pushed; redistributing raw NBA data can bump against their terms).

### 2.3 Data gotchas learned (the hard way, from real-data verification)
- **`PlayByPlayV2` is deprecated** and now returns empty JSON — must use **V3**.
- **`LeagueDashPlayerStats` returns per-game minutes**, not totals — filters must use `MIN × GP`.
- **stats.nba.com throttles** sustained scraping in ~20-minute windows — handled by per-unit resilience + resumable retry passes, not by hammering.

---

## 3. Phase 1 — the shared modeling engines

Three sub-models feed everything downstream. Built once via `python -m elevate_stat.build_models` → artifacts in `data/processed/`.

### 3.1 Win probability + leverage (`models/win_prob.py`)
A **logistic regression** predicting P(home team wins) from the game state. One training row per pbp event:
- `score_diff` = home − away (running score, forward-filled since pbp only stamps the score on scoring events),
- `seconds_remaining` = `(4 − period)·720 + clock_seconds` in regulation; `clock_seconds` in OT,
- label = did the home team win (from each game's final score).

Features: `[score_diff, √seconds_remaining, score_diff / √(seconds_remaining + 1)]`. The third term encodes "a lead matters more as time shrinks."

**Leverage index** at a state:
```
leverage(diff, secs) = | WP(diff + 2, secs) − WP(diff − 2, secs) |
```
— the win-probability swing across a one-possession (±2 pt) change. It is high late-and-close, ~0 in blowouts and early.

*Verified:* tie at 24 min ≈ 0.56 (real home-court edge); +10 at 2 min = 0.99; leverage of a tie in the final 30s = 0.68 vs **0.00** in a 20-pt blowout.

### 3.2 Expected points per shot — xPPS (`models/xpps.py`)
Basketball's "expected goals." A **HistGradientBoostingClassifier** predicts P(make) per field-goal attempt from `SHOT_DISTANCE, LOC_X, LOC_Y`, one-hot `SHOT_ZONE_BASIC`, and `is_three`. Then:
```
xPPS = P(make) × point_value      (3 if a three, else 2)
```
Two player metrics fall out: **shot-making** = actual points − Σ xPPS (finishing above expectation), and **shot-selection** = mean xPPS (do they generate good looks).

*Verified by zone:* Restricted Area **1.54** ≫ Corner-3 **1.11** > Above-break-3 **1.07** ≫ Mid-range **0.83** (textbook — this is why analytics dislikes the long two). Shot-making leaderboard tops out at Jokić / Dončić / Durant / SGA / Curry.

### 3.3 Playstyle-DNA archetypes (`models/playstyle.py`)
Unsupervised **Gaussian Mixture Model** (8 soft clusters) over standardized per-player-season features: advanced rates (`USG_PCT, AST_PCT, OREB_PCT, DREB_PCT, TS_PCT, PACE`), Synergy playtype frequencies, and three-point shot share. Produces a soft "archetype fingerprint" per player. Rotation-player filter: **total minutes = MIN × GP ≥ 500**.

*Verified:* clusters are coherent — a clear centers cluster (Drummond/Biyombo/Portis), a guards cluster (Simons/Dosunmu), etc.

---

## 4. Phase 2a — the elevation engine (the core LATE number)

### 4.1 Lineup reconstruction from play-by-play (`pbp_lineups.py`)
The hard, foundational problem: **who are the 5-on-5 on the floor at every moment?** The pbp gives substitution events but never lists period starters, so starters are *inferred*.

- **Period starters** = players who act (record any event) or are subbed *out* before they are subbed *in*, during that period.
- **Substitutions:** the row's `personId` is the player going **out**; the incoming player is parsed from the description (`"SUB: <in> FOR <out>"`) and resolved to an id.
- **Name resolution** was the real engineering. Three failure classes, all fixed via a full-name resolver:
  - *Accents* — descriptions are ASCII (`"Saric"`) but `playerName` is Unicode (`"Šarić"`) → strip diacritics.
  - *Suffixes* — `"Jackson Jr."` vs `"Jackson"` → keep the suffix as a disambiguator.
  - *Same last name* — two `Williams` on one team → disambiguate by first-initial prefix (`"Jal. Williams"`) using full names from the player-season data.
- **Validity is per-event, not per-game.** Each period re-syncs from its own starters; a bad substitution invalidates only that period's tail, and downstream keeps the good stints. **Result: ~92% of all possessions resolve to a clean 5v5.**

### 4.2 The stint dataset (`stints.py`)
A **stint** = a maximal segment with a constant 10-man lineup (breaks on any substitution or period boundary — this sidesteps fragile per-possession parsing and is the standard RAPM data prep). Per stint:
- the home & away 5-man lineups,
- `home_pts / away_pts` (score deltas, baselined to the event *entering* the stint so opening-basket points aren't dropped),
- `seconds` (clock delta),
- `leverage` (from §3.1, at the stint's entering game state),
- `player_pts` and `player_tsa` per on-court player (from Made-Shot `shotValue` / made Free-Throws; TSA = FGA + 0.44·FTA).

*Verified:* per-stint points reconcile **exactly** to final scores (e.g. 121-116), total stint time ≈ game length.

### 4.3 Leverage-weighted RAPM (`models/elevation.py`)
Regularized Adjusted Plus-Minus over the stint table via **sparse ridge regression**:
- design matrix `X` (one row per stint): `+1` for each home player, `−1` for each away player;
- response `y` = point margin per 100 possessions = `100 · (home_pts − away_pts) / possessions`, where `possessions = seconds / 28.8` (~100 team possessions per 2880s);
- two `Ridge(alpha=3000)` fits with different sample weights:
  - **`rapm`** — weight = possessions (standard adjusted plus-minus),
  - **`late`** — weight = possessions × leverage (**the leverage-aware metric**).

Coefficients = each player's points-per-100 impact, pooled across the tracking era for stability. Built via `python -m elevate_stat.build_late` → `data/processed/late_ratings.parquet` (390,754 stints, 1,544 players).

*Verified — career tracking-era RAPM top:* Curry, Jokić, Kawhi, Giannis, Embiid, Tatum, Chris Paul, Draymond, LeBron. **LATE (leverage-weighted) shifts toward clutch/playoff risers** — Kawhi jumps to #1, with Iguodala, Chris Paul, LeBron rising. That shift *is* the thesis of LATE, confirmed on real data.

---

## 5. Phase 2b/2c — teammate elevation & lenses (`models/elevation_teammate.py`)

### 5.1 The metric — WOWY on teammate *efficiency*
"Does a player make teammates better?" The first attempt measured teammate *scoring* and failed a face-validity check: it ranked low-usage shooters on top and ball-dominant stars (Embiid, Giannis) at the bottom — because raw scoring is **confounded by usage** (when a deferential player is on, others shoot more). The fix: measure teammate **efficiency**, not volume.

For teammate `B`, `scoring_rate = points/possession = (points/TSA) × (TSA/possession) = efficiency × volume`. LATE's elevation is the **efficiency** term. For every teammate pair (A, B), leverage-weighted With-Or-Without-You:
```
eff_B|A_on  = Σ(lev · B_pts while A on)  / Σ(lev · B_tsa while A on)
eff_B|A_off = (B_pts_total − with) / (B_tsa_total − with)
lift(A→B)   = 100 · (eff_B|A_on − eff_B|A_off)        # pts per 100 shot attempts
```
**Elevation-centrality(A)** = shared-attempt-weighted mean of `lift(A→B)` over all teammates B — "how much A raises teammates' efficiency."

*Verified — top elevators:* De'Aaron Fox, Curry, Jokić, Draymond, Luka, Trae Young, Haliburton, Al Horford — the creators and gravity players you'd expect. Bottom: low-impact non-creators (Alex Len, Collison).

### 5.2 The three lenses
- **Network lens** — keep the directed A→B lift matrix (not just the scalar centrality): a who-lifts-whom graph. This powers the web explorer.
- **Archetype lens** (`elevation_by_archetype`) — condition lift on the teammate's Phase-1 playstyle archetype: "does A lift shooters vs. bigs?" (e.g., Jokić lifts archetype-3 teammates +14 vs. +3 for others).
- **Mechanism lens** (`compute_mechanism`) — decompose each lift into two channels, since `Δscoring ≈ Δefficiency × Δvolume`:
  - **efficiency channel** (spacing / gravity → *better* shots),
  - **volume channel** (creation / passing → *more* shots): `100 · (TSA/poss with A − without A)`.
  Each pair lift is **shrunk** by `shared_tsa / (shared_tsa + 400)` so thin pairs don't dominate.
  *(The defense channel — easing a teammate's defensive load — needs defensive tracking we don't have, and is out of scope.)*

Built via `python -m elevate_stat.build_elevation` → `elevation_teammate.parquet`, `pairs.parquet`, `mechanism.parquet` + charts in `data/processed/figures/`.

---

## 6. The web explorer (`docs/`)

A static, GitHub-Pages-ready single page (`docs/index.html`; vis-network for the graph + hand-rolled SVG for the line chart, no build step; player headshots via `cdn.nba.com`). Data is pre-computed into `docs/data.json` by `scripts/gen_seasons.py`, which reconstructs the stint table **once** (pickle-cached), then slices it into **per-season × per-type (reg / playoffs / both) + all-time** blocks, plus a global id→name map.

**Shared controls** (apply across tabs): player search, team filter, a **reg / playoffs / both** toggle, and a **season From→To** picker (single season / range / all-time). **Three tabs:**
- **Explorer** — the player card (LATE, RAPM, elevation, the volume/efficiency mechanism split), the **who-affects-who web** (every connection, green = makes better / red = makes worse, arrow = direction, click a node to re-center), and a ranked **connections panel** (who most helps/hurts the player, toggleable to who they most help/hurt).
- **Trajectory** — a per-player **season-by-season line chart** of LATE / RAPM / Elevation (respects the reg/playoff/both toggle).
- **Leaderboards** — top elevators / LATE / RAPM, with a view-adaptive minutes threshold.

**Static-site honesty & display:** single seasons and all-time are *exact* recomputations; a custom season range is a **minutes-weighted blend** of the per-season numbers (true range re-fitting needs a server). Sub-1,000-minute totals are shown exactly (not "0k"). Because elevation is WOWY, a very high-minute star's "effect on others" can be sparse in small samples (teammates rarely play without them) — use all-time or a range there.

---

## 7. Reproduce it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python -m elevate_stat.run_ingest            # scrape -> data/raw/ (hours; resumable)
python -m elevate_stat.build_models          # win-prob, xPPS, playstyle
python -m elevate_stat.build_late            # leverage-weighted RAPM -> late_ratings.parquet
python -m elevate_stat.build_elevation       # teammate elevation + mechanism + figures
python -m scripts.gen_seasons                # -> docs/data.json for the web app
python3 -m http.server 8099 --directory docs # open http://localhost:8099
pytest                                       # 74 tests
```

---

## 8. Honest limitations

- **Teammate elevation is unadjusted WOWY (on/off), not ridge-adjusted.** A residual confound remains: role players who only ever share the floor with stars can creep up (a minutes threshold and shrinkage mitigate but don't eliminate this). A regularized teammate-residual model is genuine future work.
- **Single-season playoff webs are thin.** A ~12-game playoff run is too little shared shooting to form reliable teammate pairs — use all-time or a season range for playoff elevation.
- **~8% of possessions are dropped** where lineup reconstruction can't resolve a clean 5v5 (mostly bench players who record no event in a short stint). This is a mild, mostly-random loss.
- **Leverage/clutch splits are noisier** than season aggregates by nature; LATE pools all seasons for stability.
- **Mechanism decomposition omits defense** (no public defensive tracking), and the volume/efficiency split is an approximation of a multiplicative relationship.
- **All names/teams are best-effort** from public player-season data; a few sub-threshold connections may lack a listed team.

---

## 9. Code map

```
elevate_stat/
  config.py storage.py client.py          # ingest infrastructure (atomic, resumable, rate-limited)
  fetchers/                               # one module per nba_api endpoint family
  run_ingest.py                           # ingest orchestrator
  data.py                                 # raw-parquet loaders
  models/
    win_prob.py                           # win probability + leverage index
    xpps.py                               # expected points per shot
    playstyle.py                          # GMM playstyle-DNA archetypes
    elevation.py                          # leverage-weighted RAPM (sparse ridge)
    elevation_teammate.py                 # teammate-efficiency WOWY + mechanism + archetype lens
  pbp_lineups.py                          # on-court 5v5 reconstruction from pbp
  stints.py                               # stint dataset builder
  build_models.py build_late.py build_elevation.py   # pipeline CLIs
  viz.py                                  # matplotlib/networkx charts
scripts/
  supervise_ingest.sh                     # self-healing ingest supervisor
  gen_seasons.py export_site.py           # web-app data generation
docs/                                     # the static web explorer
tests/                                    # 74 pytest tests
```
