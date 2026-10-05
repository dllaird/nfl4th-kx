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

### Coverage
| | count |
|---|---|
| engine recommendation | 2,444 (punt 934, FG 825, go 685) |
| no recommendation | 1,553 (38.9%) |
| original nfl4th (betting lines) | go 1,655, punt 1,381, FG 961 (always recommends) |

Refusal reasons (a play can have several):

| reason | plays |
|---|---|
| outcome states outside model support | 666 |
| winner price outside [0.03, 0.97] | 535 |
| final 120 s of regulation | 250 |
| Kalshi pregame spread / total ladder unusable | 154 / 115 |
| stale price (> 30 s) | 77 |
| market wider than 4¢ | 73 |
| no price | 31 |
| halted (no trade in 60 s) | 31 |
| OT | 24 |

### Engine vs original nfl4th (2,444 plays where the engine recommends)
- **Agreement: 77.9%.** Rows are the engine's choice, columns nfl4th's:

  | engine \ nfl4th | go | FG | punt |
  |---|---|---|---|
  | go | **625** | 48 | 12 |
  | FG | 200 | **625** | 0 |
  | punt | 219 | 61 | **654** |

  The engine is **less aggressive** than nfl4th. Of 1,044 plays where nfl4th says go, the
  engine kicks 419 times.
- **Coaches' actual choices** matched the engine 82.5% of the time and nfl4th 72.5%.
- **Disagreements (540):**
  - By the engine's market valuation, its choice is worth +0.85 points more than nfl4th's.
  - By nfl4th's WP, nfl4th's choice is worth +1.13 points more than the engine's.
  - **96.7% of the engine's disagreement margins are within one standard deviation of its
    own uncertainty,** and 58% of nfl4th's margins are under 1 WP point.
  
  Most disagreements are near-ties under both methods.

### What the market did afterward
- **EV of the option actually taken vs the settled price** (3,052 plays in range):
  - Engine: mean (settled − EV) = **+0.10 points**, MAE 2.63, correlation 0.987.
  - The original nfl4th WP is not anchored to the market. Its WP of the actual choice
    differs from the settled price by −0.37 points on average (MAE 4.63).
- **By option** (engine, game-clustered 90% CI):

  | option | mean (settled − EV) | 90% CI | n |
  |---|---|---|---|
  | go | **+0.51** | [−0.02, +1.04] | 552 |
  | FG | +0.05 | [−0.17, +0.26] | 832 |
  | punt | −0.01 | [−0.13, +0.11] | 1,668 |

  When teams went for it, the market ended about half a point higher than the engine
  predicted.
  - Conversion probabilities are calibrated (realized 55.4% vs predicted 56.2%), so this
    is not coaches selecting easy attempts.
  - It is the response model's under-valuation of conversions (§1).
  - **Consequence:** the engine likely leans slightly against going for it. That is about
    the size of a typical disagreement margin.
- **Disagreement plays, by whose advice the coach followed:**
  - Followed the engine (n = 387): the market moved −0.48 points on average.
  - Followed nfl4th (n = 144): +0.17 points.
  - Difference: −0.64 points, 90% CI [−1.45, +0.21]. This is not significant, but it
    points the same way as the go-for-it bias.
  - **Caution:** this compares realized outcomes, not counterfactuals. The two groups
    differ in situation, and the market had already priced the expected choice.

### Example disagreements
Gap is the engine's EV gap ± sd, in points.

| game | situation | engine | nfl4th (go_boost) | actual | price pre → post |
|---|---|---|---|---|---|
| 2025_03_LA_PHI | Q2 0:14, 4th & 2 at 28, +9 | FG by 6.2 ± 3.0 | go (+1.3) | FG, good | 0.72 → 0.76 |
| 2025_04_LAC_NYG | Q4 5:44, 4th & 3 at 3, +3 | FG by 5.5 ± 4.2 | go (+1.6) | go, incomplete | 0.66 → 0.59 |
| 2025_21_LA_SEA | Q4 4:59, 4th & 4 at 6, −4 | FG by 4.0 ± 3.7 | go (+3.9) | go, incomplete | 0.40 → 0.25 |
| 2025_07_GB_ARI | Q4 8:05, 4th & 6 at own 20, −3 | punt by 1.8 ± 3.3 | go (+5.1) | punt | 0.37 → 0.38 |
| 2025_03_LA_PHI | Q4 12:24, 4th & 7 at 50, −5 | punt by 1.0 ± 3.7 | go (+3.5) | go, incomplete | 0.45 → 0.31 |
| 2025_16_LV_HOU | Q3 3:14, 4th & 1 at 31, −1 | go by 2.3 ± 2.2 | FG (−1.5) | FG, good | 0.76 → 0.78 |
| 2025_15_MIN_DAL | Q3 2:49, 4th & 3 at 37, −6 | go by 1.2 ± 2.3 | FG (−1.1) | go, +23 yards | 0.26 → 0.36 |

The full list is in `reports/backtest_disagreements.parquet`.

## 3. Sensitivity (2,400–2,450 recommended plays each)
| perturbation | recommendation flips | mean abs. change in go_boost |
|---|---|---|
| pre-play price −1 point / +1 point | 0.46% / 0.41% | 0.05 points |
| posteam pregame spread −1 / +1 point | 0.98% / 1.02% | 0.05 points |
| price 5 s stale | 0.25% (0.050%/s) | 0.03 points |
| price 10 s stale | 0.34% (0.034%/s) | 0.03 points |
| price 20 s stale | 0.38% (0.019%/s) | 0.04 points |
| price 30 s stale | 0.71% (0.024%/s) | 0.08 points |
| price 60 s stale | 1.64% (0.027%/s) | 0.13 points |

Recommendations are stable to small price and spread changes. A 1-point price move shifts
all options by nearly the same amount, so it rarely reorders them. Staleness of up to 30 s
flips fewer than 1% of decisions, which supports the 30 s default.

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
- **Fixed components:** the conversion, FG and punt components are nfl4th's, not refit.
  They don't see in-game changes (backup QB, injured kicker), and team strength is the
  Kalshi pregame line, fixed for the game.
