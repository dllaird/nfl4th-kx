# RESPONSE_MODEL — predicting post-play Kalshi winner prices (Phase 5)

Code:
- `nfl4th_py/market/response.py` — the models
- `scripts/train_response.py` — training and evaluation
- `scripts/analyze_response.py` — diagnostics

The trained models live in `nfl4th_py/data/response/` (all 279 games with Kalshi pregame
lines). Metrics are in `reports/response_metrics.json`.

## Data
41,183 transitions from the Phase 4 dataset: usable plays with both Kalshi pregame lines
and a known next state. They cover all play types, not only 4th downs. Each row is from
the **pre-play possession team's** view:
- **pre-play state** + pre-snap price (logit) + Kalshi pregame spread/total;
- **post-play state:** who has the ball; field position from the team's scoring end zone;
  score; down and distance; clock; timeouts; and whether the next snap is a scrimmage
  down, a PAT or a kickoff.

**Target:** the logit of the settled post-play price.

## Formulations
| | description |
|---|---|
| **(a) transition** | xgboost with base margin = pre-snap logit, so the model learns the *change*. Features: pre state, post state and deltas (score, field position, clock). |
| **(b) value surface** | xgboost V(state, Kalshi spread, total) fit to observed pre-snap prices (~43k states). Counterfactual = V(post) re-anchored by the offset logit(price) − V(pre-state). An opponent-possession state is valued as −V from the opponent's side. |
| persistence | post = pre (the market doesn't move) |
| wp_baseline | the nfl4th WP models, with the home spread solved per play so the pre-play WP equals the market price (solved for 96.2% of rows), then applied to the post state. Only defined for scrimmage pre-states. **Benchmark only.** |

**Monotone constraints:**
- (a): post field position, post score, possession, and their deltas;
- (b): field position, score and team spread.

## Validation
All splits are by game, and every comparison uses identical rows.
- **Time holdout:** train weeks 1–14 (15% of those games held for early stopping), test
  weeks 15–22 (late season + playoffs).
- **Grouped 5-fold CV** over all games, so every 4th down gets an out-of-sample
  prediction.

Each cell below is logit RMSE / price MAE in cents. Δ wp is transition minus
wp_baseline in logit RMSE, with a game-clustered bootstrap 90% CI. Negative means the
transition model is better. All segments are restricted to pre-snap prices in
[0.03, 0.97].

### Grouped 5-fold CV (all games)
| segment | n | persistence | **transition** | value | wp_baseline | Δ wp [90% CI] |
|---|---|---|---|---|---|---|
| all plays | 32,883 | 0.274 / 2.52 | **0.129 / 1.08** | 0.157 / 1.35 | 0.168 / 1.46 | −0.038 [−0.042, −0.034] |
| all 4th-down plays | 3,321 | 0.409 / 3.02 | **0.185 / 1.44** | 0.233 / 2.21 | 0.231 / 2.16 | −0.046 [−0.060, −0.033] |
| 4th down converted | 353 | 0.541 / 6.73 | **0.222 / 2.09** | 0.249 / 2.46 | 0.257 / 2.48 | −0.035 [−0.084, +0.009] |
| 4th down failed | 277 | 0.987 / 7.76 | **0.311 / 1.68** | 0.410 / 1.96 | 0.384 / 2.28 | −0.073 [−0.122, −0.025] |
| FG made | 781 | 0.285 / 1.98 | **0.186 / 1.40** | 0.215 / 2.23 | 0.233 / 2.30 | −0.046 [−0.084, −0.006] |
| FG missed/blocked | 125 | 0.739 / 8.53 | 0.325 / 2.35 | **0.287 / 2.44** | 0.334 / 2.96 | −0.008 [−0.041, +0.038] |
| 4th-down punts | 1,785 | 0.167 / 1.62 | **0.128 / 1.23** | 0.192 / 2.17 | 0.178 / 1.95 | −0.050 [−0.062, −0.036] |
| touchdowns | 1,220 | 0.497 / 5.88 | **0.183 / 1.57** | 0.262 / 2.78 | 0.308 / 3.16 | −0.125 [−0.147, −0.105] |
| turnovers | 549 | 0.945 / 11.33 | **0.298 / 2.23** | 0.353 / 2.14 | 0.337 / 2.26 | −0.039 [−0.065, −0.013] |
| OT | 237 | 0.571 / 7.36 | 0.315 / 3.43 | **0.296 / 3.41** | 0.443 / 4.48 | −0.128 [−0.183, −0.070] |
| last 2 min, within 8 pts | 959 | 0.938 / 8.01 | **0.435 / 3.61** | 0.523 / 4.29 | 0.541 / 4.45 | −0.106 [−0.137, −0.072] |

### Time holdout (weeks 15–22)
| segment | n | persistence | **transition** | value | wp_baseline | Δ wp [90% CI] |
|---|---|---|---|---|---|---|
| all plays | 9,063 | 0.270 / 2.52 | **0.130 / 1.05** | 0.150 / 1.31 | 0.158 / 1.42 | −0.027 [−0.036, −0.017] |
| all 4th-down plays | 928 | 0.395 / 3.09 | **0.196 / 1.42** | 0.229 / 2.23 | 0.221 / 2.07 | −0.025 [−0.068, +0.027] |
| 4th down converted | 107 | 0.528 / 7.28 | **0.211 / 2.05** | 0.212 / 2.47 | 0.226 / 2.34 | −0.015 [−0.064, +0.027] |
| 4th down failed | 83 | 0.851 / 7.24 | **0.269 / 1.93** | 0.324 / 2.01 | 0.323 / 2.22 | −0.054 [−0.124, +0.035] |
| FG made | 235 | 0.388 / 2.20 | 0.283 / 1.40 | 0.268 / 2.18 | **0.252** / 2.10 | +0.032 [−0.109, +0.125] |
| FG missed/blocked | 32 | 0.562 / 8.42 | 0.167 / 2.43 | **0.163 / 2.18** | 0.179 / 2.59 | −0.012 [−0.062, +0.036] |
| 4th-down punts | 471 | 0.141 / 1.49 | **0.104 / 1.14** | 0.191 / 2.24 | 0.181 / 1.94 | −0.077 [−0.101, −0.053] |

**Choice: (a), the transition model.** It is best or tied in nearly every segment, and it
is the only formulation that clearly beats persistence on punts (b and wp_baseline are
worse than "no move" there). It beats wp_baseline overall on both splits.
- **Where it does not clearly win:** on the 4th-down sub-segments in the late-season
  holdout, the CIs include 0. On holdout FG makes, its logit RMSE is worse than
  wp_baseline's, though its price MAE is better (1.40¢ vs 2.10¢) and the CI is wide
  (n = 235).
- **Where (b) is competitive:** FG misses and OT.

## Diagnostics (out-of-fold, transition model)
- **Price calibration:** predicted and actual post prices agree within 0.1¢ in every bin
  from 0–5% to 95–100%.
- **Uncertainty:** an xgboost model of |residual|, rescaled by 1.206 so that ±1.645 sd
  covers 90% of out-of-fold residuals. After rescaling, coverage is 71.7% at ±1 sd,
  90.0% at ±1.645 sd and 93.9% at ±1.96 sd. The scale was set on the same out-of-fold
  residuals it is checked on, so this is calibration, not an independent test.
- **Support:** the mean distance to the 25 nearest training transitions in a standardized
  key-feature space, mapped to (0, 1].
  - The 10% least-supported predictions have 2.6× the error of the best-supported half
    (MAE 0.155 vs ~0.06 logit).
  - 12% of transitions have support < 0.5 (MAE 0.15).
- **Edge cases:**
  - **Extreme prices** (pre-snap < 0.03 or > 0.97; 11% of plays): **the transition model
    is worse than persistence** (MAE 0.175 vs 0.125 logit). **The decision engine must
    refuse there.**
  - **OT and close end-of-game situations:** the model's errors are 2.4–3.4× the
    in-range average (logit RMSE 0.32 and 0.44 vs 0.13), but it still beats every
    alternative except (b) in OT. Expect wide uncertainty.
  - **Price source:** errors are similar whether a price came from a trade VWAP or the
    last trade. Coarse-timestamp plays are slightly worse (0.192 vs 0.129) but still beat
    the baseline.
- **Property tests** (`tests/test_response_properties.py`, 2,000 synthetic states each):
  better field position, a bigger lead and possession never lower the prediction, for
  both (a) and (b).

## Caveats
- **One season** of data. The 4th-down segments have tens to hundreds of plays, so most
  4th-down differences against wp_baseline are not significant on the late-season holdout.
- **wp_baseline's comparison is generous:** it is anchored with a spread solved per play,
  which is itself market information. It still loses overall.
- **The model learns realized transitions.** Its counterfactual use for 4th downs (Phase 6)
  assumes the market would price a hypothetical outcome the way it priced similar real
  ones. The anchoring question is discussed in DECISION.md.
