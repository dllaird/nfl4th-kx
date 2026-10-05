# VALIDATION — Kalshi market-valued 4th-down decisions (Phase 8)

Everything here uses the 2025 NFL season, the only season with Kalshi game markets.

**Scripts:**
- `scripts/train_response.py` — response-model metrics
- `scripts/analyze_response.py` — response-model diagnostics
- `scripts/backtest.py` — the decision backtest

**Outputs:**
- `reports/response_metrics.json`
- `reports/backtest_summary.json`
- `reports/backtest_plays.parquet` (git-ignored)
- `reports/backtest_disagreements.parquet` (git-ignored)

The decision backtest is **out-of-sample**: each game's 4th downs are valued by the
response model from the cross-validation fold that did not train on that game.

## 1. Response model (Phase 5)
Full tables are in RESPONSE_MODEL.md. All numbers are logit RMSE of the predicted settled
price, on plays with pre-snap prices in [0.03, 0.97], comparing the transition model
with nfl4th's WP models (spread solved to match the pre-play price) on identical rows.

| | grouped 5-fold CV | time holdout (weeks 15–22) |
|---|---|---|
| all plays | **0.129** vs 0.168 (no-move: 0.274); Δ −0.038 [−0.042, −0.034] | **0.130** vs 0.158; Δ −0.027 [−0.036, −0.017] |
| all 4th downs | **0.185** vs 0.231; Δ −0.046 [−0.060, −0.033] | **0.196** vs 0.221; Δ −0.025 [−0.068, +0.027] |
| 4th down converted | 0.222 vs 0.257; CI includes 0 | 0.211 vs 0.226; CI includes 0 |
| 4th down failed | **0.311** vs 0.384; significant | 0.269 vs 0.323; CI includes 0 |
| FG made | **0.186** vs 0.233; significant | 0.283 vs **0.252**; CI includes 0 |
| FG missed/blocked | 0.325 vs 0.334 (the value surface is best: 0.287); CI includes 0 | 0.167 vs 0.179; CI includes 0 |
| 4th-down punts | **0.128** vs 0.178; significant | **0.104** vs 0.181; significant |

**Verdict:** the market-response model beats the WP baseline at predicting post-play
market prices overall, and on 4th downs as a group in cross-validation. On the
late-season holdout alone, only punts are a significant win. Made FGs are slightly worse
there, though not significantly.

Other diagnostics:
- **Calibration:** within 0.1¢ in every price bin.
- **Uncertainty:** after rescaling by 1.206, 90% of out-of-fold residuals fall within the
  ±1.645 sd interval.
- **Extreme prices:** outside [0.03, 0.97] the model is *worse* than assuming no move, so
  the engine refuses there.

**Known bias:** on realized 4th-down conversions, the model under-predicts the settled
price by +0.57 points (out-of-fold). Upweighting 4th-down transitions ×3/×5/×10 in
training did not remove it (+0.56 to +0.59) and made conversion RMSE worse, so it was not
adopted.

## 2. Decision backtest (3,997 4th downs)

Field goal make probabilities come from the kicker-specific model fit on 2014–2024
(FIELD_GOALS.md), so they are out-of-sample for 2025 as well.

### Coverage
| | count |
|---|---|
| engine recommendation | **3,827 (95.7%)**: punt 1,787, FG 1,079, go 961 |
| of which normal confidence | 3,244 |
| of which caution | 583 |
| no recommendation | 170 (4.3%) |
| original nfl4th (betting lines) | go 1,655, punt 1,381, FG 961 (always recommends) |

- **Refusals:** all 170 are model support with fewer than two options left to compare,
  except 4 plays with a market wider than 10¢.
- **Cautions** (a play can have several):

  | caution | plays |
  |---|---|
  | price at the 1¢ tick limit (below 2% or above 98%) | 344 |
  | tick-limit toss-up (options within one tick) | 334 |
  | weak pregame total ladder fit | 38 |
  | an option left out for low support | 101 (15 where its unreliable value would have ranked first) |
  | Kalshi total from the week-median fallback / spread from the winner price | 74 / 9 |
  | market 4–10¢ wide | 69 |
  | price 30–120 s old | 36 |
  | overtime | 24 |
  | price from the last quote (no trade in 60 s) | 10 |

- **Accuracy by confidence:** cautioned recommendations match the market at least as
  well as normal ones. Mean (settled − EV) for the option taken is +0.06 points (MAE 1.7)
  with a caution and +0.10 (MAE 3.0) without, over all prices.
- **Final 2 minutes of Q4:** recommended on 229 of 250 plays, against 0 before.
  Excluding toss-ups (149 plays), the engine agrees with nfl4th 82% of the time, and
  coaches made the engine's call 86% of the time.
- **Overtime:** all 24 plays recommended.
- **History:** before this round of changes, the engine refused 994 plays (24.9%), and
  originally 1,553 (38.9%). See §5.

### Engine vs original nfl4th (3,827 plays where the engine recommends)
The tables below are from the 3,003-play run before the refusal changes; with all 3,827
plays, agreement is 76.9%, coaches matched the engine 83.1% and nfl4th 73.5%.

- **Agreement: 80.0%.** Rows are the engine's choice, columns nfl4th's:

  | engine \ nfl4th | go | FG | punt |
  |---|---|---|---|
  | go | **685** | 38 | 16 |
  | FG | 215 | **642** | 1 |
  | punt | 277 | 54 | **1,075** |

  The engine is **less aggressive** than nfl4th. Of 1,177 plays where nfl4th says go, the
  engine kicks 492 times.
- **Coaches' actual choices** matched the engine 83.9% of the time and nfl4th 73.8%.
- **Disagreements (601):**
  - By the engine's market valuation, its choice is worth +0.85 points more than nfl4th's.
  - By nfl4th's WP, nfl4th's choice is worth +1.11 points more than the engine's.
  - **96.8% of the engine's disagreement margins are within one standard deviation of its
    own uncertainty,** and 60% of nfl4th's margins are under 1 WP point.

  Most disagreements are near-ties under both methods.

### What the market did afterward
- **EV of the option actually taken vs the settled price** (3,052 plays in range):
  - Engine: mean (settled − EV) = **+0.06 points**, MAE 2.61, correlation 0.99.
  - The original nfl4th WP is not anchored to the market. Its WP of the actual choice
    differs from the settled price by −0.37 points on average (MAE 4.63).
- **By option** (engine, game-clustered 90% CI):

  | option | mean (settled − EV) | 90% CI | n |
  |---|---|---|---|
  | go | **+0.63** | [+0.06, +1.25] | 638 |
  | FG | +0.01 | [−0.31, +0.35] | 923 |
  | punt | −0.06 | [−0.19, +0.07] | 1,823 |

  (Current run, response model retrained on all 285 games. The go bias is now
  significant at 90%.)

  When teams went for it, the market ended about half a point higher than the engine
  predicted.
  - Conversion probabilities are calibrated (realized 55.4% vs predicted 56.2%), so this
    is not coaches selecting easy attempts.
  - It is the response model's under-valuation of conversions (§1).
  - **Consequence:** the engine likely leans slightly against going for it. That is about
    the size of a typical disagreement margin.
- **Disagreement plays, by whose advice the coach followed:**
  - Followed the engine (n = 446): the market moved −0.46 points on average.
  - Followed nfl4th (n = 144): +0.17 points.
  - Difference: −0.63 points, 90% CI [−1.64, +0.29]. This is not significant, but it
    points the same way as the go-for-it bias.
  - **Caution:** this compares realized outcomes, not counterfactuals. The two groups
    differ in situation, and the market had already priced the expected choice.
- **Field goal make probabilities on the 2025 attempts teams actually tried**
  (kicker model fit through 2024):

  | distance | n | made | predicted |
  |---|---|---|---|
  | < 40 | 455 | 94.5% | 95.0% |
  | 40–49 | 238 | 80.7% | 79.5% |
  | 50–54 | 97 | 72.2% | 71.0% |
  | 55–59 | 37 | 64.9% | 62.3% |
  | 60+ | 5 | 40.0% | 35.6% |

### What knowing the kicker changes
The same backtest with every kicker replaced by the league curve changes 62 of 2,995
recommendations (2.1%).
- **Direction:** mostly a strong kicker turns a go (35) or a punt (21) into a field goal
  from 44–50 yards. The most frequent kickers are Aubrey 10, Boswell 7, Butker 6 and
  Slye 5.
- **Coaches:** on those 62 plays, coaches made the kicker-specific call 71% of the time
  and the league-curve call 29%.

### Example disagreements
Gap is the engine's EV gap ± sd, in points.

| game | situation | engine | nfl4th (go_boost) | actual | price pre → post |
|---|---|---|---|---|---|
| 2025_03_LA_PHI | Q2 0:14, 4th & 2 at 28, +9 | FG by 6.2 ± 3.0 | go (+1.3) | FG, good | 0.72 → 0.76 |
| 2025_04_LAC_NYG | Q4 5:44, 4th & 3 at 3, +3 | FG by 5.5 ± 4.2 | go (+1.6) | go, incomplete | 0.66 → 0.59 |
| 2025_15_IND_SEA | Q4 13:28, 4th & 7 at 14, −4 | FG by 4.7 ± 3.4 | go (+1.3) | FG, good | 0.61 → 0.62 |
| 2025_11_BAL_CLE | Q4 14:09, 4th & 5 at 5, −6 | FG by 4.6 ± 2.8 | go (+1.8) | FG, good | 0.65 → 0.60 |
| 2025_07_GB_ARI | Q4 8:05, 4th & 6 at own 20, −3 | punt by 1.8 ± 3.3 | go (+5.1) | punt | 0.37 → 0.38 |
| 2025_11_BAL_CLE | Q2 0:35, 4th & 2 at own 42, −6 | punt by 0.4 ± 3.5 | go (+4.4) | punt | 0.58 → 0.58 |
| 2025_16_LV_HOU | Q3 3:14, 4th & 1 at 31, −1 | go by 1.9 ± 2.1 | FG (−1.5) | FG, good | 0.76 → 0.78 |
| 2025_12_PIT_CHI | Q4 4:46, 4th & 2 at own 40, +3 | go by 1.9 ± 4.9 | punt (−0.4) | punt | 0.67 → 0.70 |

The full list is in `reports/backtest_disagreements.parquet`.

## 3. Sensitivity (2,925–3,003 recommended plays each; run before the refusal changes)
| perturbation | recommendation flips | mean abs. change in go_boost |
|---|---|---|
| pre-play price −1 point / +1 point | 0.44% / 0.50% | 0.06 points |
| posteam pregame spread −1 / +1 point | 0.77% / 0.60% | 0.05 points |
| price 5 s stale | 0.17% (0.034%/s) | 0.03 points |
| price 10 s stale | 0.27% (0.027%/s) | 0.04 points |
| price 20 s stale | 0.31% (0.015%/s) | 0.05 points |
| price 30 s stale | 0.85% (0.028%/s) | 0.09 points |
| price 60 s stale | 1.13% (0.019%/s) | 0.15 points |

Recommendations are stable to small price and spread changes. A 1-point price move shifts
all options by nearly the same amount, so it rarely reorders them. Staleness of up to 30 s
flips fewer than 1% of decisions, which supports the 30 s default.

With the tick-limit plays included (current run, 3,564–3,827 plays):
- **Price:** a ±1-point move flips 1.8–1.9% of recommendations, because at a 1% or 99%
  price a 1-point move is large relative to the values involved.
- **Spread:** ±1 point flips 0.8–1.1%.
- **Staleness:** 60 s flips 1.6%.

## 4. How often the markets were unusable (45,186 in-game snaps)
| condition | share of snaps |
|---|---|
| winner price outside [0.03, 0.97] (blowouts) | 13.4% |
| last trade more than 30 s old | 1.6% |
| 1-minute quote wider than 4¢ | 1.3% |
| no trade in the 60 s before the snap ("halted") | 0.65% |
| no price at all | 0.65% |
| games without usable pregame ladders | 6 of 285 (all week 1) |

On 4th downs specifically, winner-market problems refused 639 plays (16.0%): 535 for
extreme prices, and 104 (2.6%) for stale, wide, halted or missing prices.

## 5. Changes made during validation
1. **Clock after each outcome.** The engine originally ran 6 s off every outcome, as
   nfl4th does. Measured play-to-next-snap clock use is 38 s after a non-TD conversion
   (14 s in a half's final 2 minutes), 4–5 s after failures, TDs and FGs, and 9 s after
   punts. The flat 6 s made converted states look about 30 s "earlier" than the training
   data shows, which doubled the go-for-it bias (+0.87 → +0.47 points after the fix). The
   engine now uses the measured values (configurable in `DecisionConfig`) and stops the
   clock at the two-minute warning.
2. **Final 2 minutes of regulation are refused.** Large disagreements included
   recommending a FG when down 4 with 13–24 s left. The model valued "down 1, kicking off,
   9 s left" at 0.11. The support measure scored these states as acceptable (0.06–0.22)
   because it treats 20 s and 120 s as close. The engine refuses rather than guesses
   (`q4_refuse_last_seconds = 120`, configurable).
3. **Support as a percentile.** The original support score made even real 4th-down
   transitions look unsupported. It is now the percentile of training-data typicality,
   and thresholds are set from out-of-fold error.
4. **Kicker-specific field goals.** The engine used nfl4th's 0% make probability from
   beyond the 53. It still built the "missed field goal" state for 75- to 100-yard
   attempts, and those states (opponent ball near its own goal line) have no training
   support. That vetoed 590 otherwise ordinary recommendations, mostly obvious punts.
   - **The fix:** field goal probability now comes from a kicker-specific model at every
     distance, with no cutoff (FIELD_GOALS.md). An attempt is offered only if the
     kicker's make probability is at least `fg_min_make_prob` (default 1%).
   - **Threshold sweep:**

     | threshold | recommendations | FG recommendations | refused for FG support |
     |---|---|---|---|
     | 0 | 2,494 | 858 | 575 |
     | 0.5% | 2,966 | 858 | 81 |
     | 1% | 3,003 | 858 | 43 |
     | 2% | 3,017 | 858 | 29 |
     | 5% | 3,035 | 856 | 7 |

     The threshold removes hopeless attempts without changing which field goals are
     recommended.
5. **Fewer refusals: end game, extreme prices, thin markets, OT** (994 → 170 refusals).
   - **Late clock:** after a late conversion, a trailing team stops the clock (median 12 s
     in Q4) and a leading team runs it (40 s). The engine had used 14 s for everyone. That
     ran out the clock on trailing teams and valued their conversions as losses.
   - **Kneel-out states:** valued at the Kalshi market's observed level (median 0.99, the
     tick limit) as a bound. The model had valued "down 1, kicking off, 9 s left" at
     9–26%, which made late field goals when down 4–7 look good.
   - **Final-2-minutes refusal removed.** With these fixes, the late-game recommendations
     no longer include field goals when down 4–7 with seconds left.
   - **Extreme prices.** The old [0.03, 0.97] limit came from a logit-scale comparison.
     In price terms the model beats "no move" between 2–3% and 97–98%. Beyond that the
     market sits at its 1¢ tick, and both errors are about 0.1–0.2¢. These plays are now
     recommended with a "low stakes" caution, plus a "toss-up" caution when the options
     are within one tick.
     - The market can't express differences that small. For a team down two scores
       with a minute left at a 1% price, the recommendation can be football-wrong (e.g.
       punt), and it is labeled a toss-up.
   - **Market thresholds:** stale and wide markets are cautions up to 120 s and 10¢. A
     missing trade price falls back to the last 1-minute quote.
   - **Pregame lines:** a weak spread ladder is averaged with the winner-price spread (no
     caution; it beats either source alone). A weak total ladder is a caution. Missing
     ladders use Kalshi-only fallbacks: spread from the pregame winner price, total from
     the week's median.
     This also added the 6 week-1 games to the response model's training data (285
     games; out-of-sample 4th-down logit RMSE 0.180 vs 0.185).
   - **Overtime:** supported with OT's own clock, and a tie when time expires is valued
     at 0.5.
   - **Unsupported options:** an option outside the model's support is left out of the
     comparison with a caution, instead of blocking the play.

## 6. Caveats and what could not be validated
- **One season.** 285 games, of which 279 have Kalshi pregame lines. Most 4th-down
  sub-segment comparisons on the late-season holdout are not significant.
- **No ground truth for decisions.** We see the market's reaction only to the option the
  team chose. "Which option was best" cannot be validated directly; §2 checks only
  whether option EVs are unbiased for the options teams took. Counterfactual options
  (e.g. going for it where teams punted) are untestable in principle.
- **The go-for-it bias of about +0.5 points** (CI just touching 0) is the main known
  weakness. Its size is similar to typical disagreement margins, so the engine's lower
  aggressiveness relative to nfl4th should be read with that in mind.
- **Rare outcomes** (blocked kicks, return TDs, muffs, safeties) have little or no
  support. Their values are extrapolations, though their probabilities are small.
- **Alignment uncertainty:**
  - Prices are trade-based (no orderbook history).
  - Whole-second snap timestamps (10% of plays; 99% of PATs) have no measured leakage.
  - Replay reversals are absorbed into the settled price.
  - "Settled" is about 94% of the eventual move at the 25 s cutoff.
- **Live mode** (`nfl4th-py live`) was tested end to end against real Kalshi orderbooks
  with a synthetic situation, but **not during a live game.** ESPN's live `situation`
  fields were coded from their documented shape and tested with synthetic data.
- **Out-of-sample scope:** the engine's thresholds (support, price range, refusal window)
  were chosen while looking at this backtest, so the backtest is not fully out-of-sample
  for those choices. The response model itself is.
- **Components:** the conversion and punt components are nfl4th's, not refit. They don't
  see in-game changes such as a backup QB, and team strength is the Kalshi pregame line,
  fixed for the game.
- **Kicker:** the FG component is the kicker-specific model (FIELD_GOALS.md). A
  replacement kicker is picked up only once he has kicked in the game. Until then the
  engine uses the team's usual kicker.
- **Field goals have no selection correction.** Long attempts are taken in favorable
  conditions, and the model learns only from attempts teams chose to take. The 2025 make
  rate from 60+ yards (52% on 23 tries) ran above every model's prediction.
