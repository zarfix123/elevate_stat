# LATE — Leverage-Adjusted Teammate Elevation

**Working name:** LATE *(Leverage-Adjusted Teammate Elevation — and yes, the pun is intentional: leverage-weighting emphasizes late-game, high-stakes possessions).*
**Status:** Design approved — pre-implementation
**Date:** 2026-07-03
**Scope for v1:** Last decade — **2015–16 season → present** (11 completed seasons through 2025–26). Start point chosen so *every* season carries the identical full feature set (Synergy playtypes begin 2015–16): consistency over raw length.
**Data source:** Public only — `nba_api` (stats.nba.com endpoints), with basketball-reference / Kaggle as backfill/validation.

---

## 1. The one-liner

**How much does a player raise their teammates' production *above those teammates' own expected baseline*, per 100 possessions, weighted toward the moments that actually decide games?**

Existing "makes teammates better" measures (WOWY, on/off, gravity proxies, Box Creation) share two weaknesses: they are **raw** (correlation, not lift) and **possession-blind** (a blowout counts the same as a tie game with 40 seconds left). LATE attacks both.

## 2. Why it's novel (honest prior-art read)

The *concept* of teammate elevation is well-worked: with-or-without-you (WOWY), on/off splits, BBall-Index's Box Creation / Passer Rating / Offensive Load, the fuzzy idea of "gravity," and 2-/3-man lineup net ratings. What has **no clean public version**:

- **Counterfactual framing** — measuring lift against each teammate's *own* expected baseline instead of raw shared-floor production.
- **Leverage-weighting** — standard in baseball (WPA/Leverage Index), essentially absent from public basketball elevation work.
- **A directed who-lifts-whom network**, mechanism decomposition, and archetype-conditional lift, all off one engine.

These are the seams we own.

## 3. Core computation (the engine — built once)

For each player **X**, over possessions where X is on the floor:

1. For every teammate **T** sharing the floor, compute T's **actual points produced** and T's **expected baseline** on those possessions (baseline from the xPPS model, §4.2).
2. The residual `actual − expected` is teammate production *above what T would be expected to do on their own*.
3. Attribute that residual to X — but **not** raw, because "with X on" is confounded by the other four players. We estimate it as a **regularized (ridge) regression of teammate residuals on the on-court indicators of candidate elevators**, with **possession weights = leverage**. This is, in spirit, "RAPM run on teammate residuals instead of raw margin."
4. Output is expressed **per 100 possessions, leverage-weighted**.

**The honest hard part:** attribution — was it X who lifted T, or a third teammate also on the floor? This is the same confounding RAPM fights. v1 uses on/off pairs + ridge regularization to disentangle it and is **explicit that the estimate is approximate**. Being upfront about this is the intellectual core of the project.

**Scope of v1:** LATE measures **offensive elevation** — teammate *scoring* production above baseline. Defensive elevation is a genuinely separate (and harder) problem; v1 touches it only as the "defensive relief" proxy channel in the mechanism lens (§5.2), not as part of the core number.

## 4. Shared sub-models (each is one of the researched "alternative" ideas, repurposed)

### 4.1 Win-Probability / Leverage model
Logistic model on public play-by-play (score margin, time remaining, possession, team-strength prior). **Leverage** of a possession = sensitivity of win probability to that possession's outcome (baseball's Leverage Index, adapted). High late-and-close, ~0 in garbage time. *Also powers the momentum stretch-goal (§9).*

### 4.2 Expected-Points (xPPS) model — the baseline engine
Gradient-boosted model predicting expected points per shot from location (LOC_X/Y, distance, zone), shot type, and tracking context (catch-and-shoot vs. pull-up, dribbles, touch time, closest-defender bucket). Gives each teammate's expected efficiency **given the shots they actually take** — the counterfactual baseline. *Caveat: NBA changed tracking-data segmentation post-2020; handle the pre/post-2020 split explicitly.*

### 4.3 Playstyle-DNA clustering — archetype labels
Unsupervised clustering (GMM for soft/mixture membership) over per-player rate stats + shot-distribution + Synergy playtype mix, producing teammate **archetype labels** used by the archetype lens (§5.3). Only the well-trodden labeling part is needed here. *(Synergy playtypes start 2015–16, which is exactly why v1 begins there — the whole window has consistent features.)*

## 5. The three lenses (all read off the same engine's output)

### 5.1 Network lens
Keep the pairwise **A → B** lift estimates instead of collapsing X to one scalar. Yields a directed, weighted **who-lifts-whom graph**:
- **Elevation centrality** = net lift a player gives others (weighted out-degree).
- **Overperforming duos** = pairs with the largest positive residual (chemistry).

### 5.2 Mechanism lens
Decompose each lift into **how** it happens:
- **Shooting / spacing (gravity):** teammate's *shot quality* (xPPS) rises with X on.
- **Passing / creation:** teammate gets *more* or *easier* shots (higher assisted rate / attempt volume).
- **Defensive relief:** teammate's defensive workload / matchup difficulty eases. *(Hardest to measure on public data — may be proxy-only in v1; flagged as such.)*

### 5.3 Archetype lens
Condition elevation on teammate archetype (from §4.3): e.g. "+6 to catch-and-shoot wings, +1 to bigs." Actionable for roster construction.

## 6. Data scope & manifest (2015–16 → present)

| Purpose | `nba_api` endpoint(s) | Notes |
|---|---|---|
| Play-by-play (leverage, on/off, lineups) | `PlayByPlayV3` | V2 deprecated (returns empty JSON); V3 adds inline shot coords + running score. Reconstruct 5-on-5 via substitution events |
| Shots (location) | `ShotChartDetail`, `LeagueDashPlayerShotLocations` | LOC_X/Y, distance, zone, made flag |
| Tracking shot detail (baseline features) | `LeagueDashPlayerPtShot` | League-wide per-player, split by closest-defender bucket (~4 calls/season vs. ~500 per-player). Post-2020 segmentation caveat |
| Lineups / on-off | `LeagueDashLineups`, `BoxScoreAdvancedV2` | Cross-check against PBP-reconstructed lineups |
| Playtypes (archetype features) | `SynergyPlayTypes` | 2015–16+ |
| Player-season stats (clustering) | `LeagueDashPlayerStats` (Base/Advanced/Scoring/Usage) | |

**Backfill/validation:** DomSamangy `NBA_Shots` dataset and basketball-reference / Kaggle mirrors.

## 7. Data pipeline (start early, but scrape once — correctly)

`nba_api` is **rate-limited and flaky**; ~11 seasons of shot-logs + play-by-play + lineups is *tens of thousands* of requests / multi-hour. So the pull is **not** a blind agent blast — it's a **resumable, checkpointed, rate-limited pipeline**:
- Per-season / per-game checkpointing so it resumes after failures.
- Polite rate-limiting + retry/backoff + rotating request headers.
- Raw responses cached to disk (parquet) so we never re-pull.
- This is **build step 1** and runs in the background while the models are coded.

## 8. Validation (this is what separates a legit stat from a toy)

- **Out-of-sample stability:** does first-half-season LATE predict second-half LATE? (RAPTOR-style in-/out-of-sample test — drop anything that only fits in-sample as luck.)
- **Predictive lift:** does LATE improve prediction of lineup / team offensive rating *beyond* raw on/off?
- **Face validity:** do known elevators (Jokić, LeBron, CP3, Curry, Draymond) top the board, and "empty-stats" players rank low?
- **Retrodiction:** does adding LATE improve team-outcome models vs. existing public impact metrics?

## 9. Roadmap

- **Phase 0 — Data pipeline** (§7), tracking era.
- **Phase 1 — Shared engines:** win-prob/leverage, xPPS baseline, DNA clusters.
- **Phase 2 — Core elevation engine:** leverage-weighted, regularized (§3).
- **Phase 3 — Three lenses:** network, mechanism, archetype (§5).
- **Phase 4 — Visualizations** (§10).
- **Phase 5 (later):** extend back toward 30 seasons (reduced, location-only baseline pre-2013); momentum "Spark/Stabilize" stretch-goal (runs on the §4.1 engine).

## 10. Visualizations (the fun part)

- **Elevation network** — force-directed, arrows weighted by lift.
- **Who-lifts-whom heatmap** — roster matrix.
- **Career-trajectory elevation line** — LATE season by season (the "develop over time" view; honest across history because leverage is structural, not recency-based).
- **Mechanism stacked-area** — how a player's elevation is composed over time.
- **Archetype bar** — elevation by teammate type.
- **League scatter** — elevation centrality vs. raw impact.

## 11. Tech stack (proposed)

Python · `nba_api` · pandas/numpy · scikit-learn (ridge, gradient boosting, GMM) · possibly statsmodels · `networkx` for the graph · matplotlib/plotly for viz · parquet for storage.

## 12. Open questions & risks

- **Attribution confounding** (§3) — the core methodological risk; regularization mitigates but does not eliminate.
- **`nba_api` rate limits / stability** — mitigated by the resumable pipeline (§7).
- **Small-sample duos** — low shared-minute pairs are noisy; needs a minutes threshold / shrinkage.
- **Defensive-relief channel** (§5.2) — may only be a proxy with public data.
- **Tracking segmentation change post-2020** — must be handled in the xPPS model.
- **Not yet under version control** — no git repo initialized for this project.
