# FIELD_GOALS — kicker-specific make probability

Code:
- `nfl4th_py/models/kicker.py` — the model
- `scripts/export_kicks.R` — exports the kick data
- `scripts/train_kicker.py` — fitting and validation; metrics in `reports/kicker_metrics.json`

The model lives in `nfl4th_py/data/kicker_model.json`, fit on 2014–2025.

The market decision engine uses this model. The nfl4th port (`models/field_goal.py`) and
the `wp_baseline` benchmark keep nfl4th's roof/era GAM, so parity with R is unchanged.

## Why
nfl4th's field goal model has no kicker, and it sets make probability to 0% from the
opponent's 53 out (kicks of 70+ yards), with a linear decay from 58 yards. In the market
engine, that cutoff caused two problems:
1. **Every kicker was treated as average.** A strong-legged kicker and a weak one had
   the same 55-yard odds.
2. **Hopeless attempts still blocked decisions.** A 0% field goal from a team's own 15
   was still built as an option, and its "miss" state (opponent ball near its own goal
   line) is outside the response model's data. That blocked 590 recommendations in the
   2025 backtest.

## Model
A logistic regression with partial pooling, fit by maximum a posteriori on 28,147 kicks
(12,787 field goals, the rest extra points) by 125 kickers, 2014–2025:

```
logit P(make) = spline(distance) + outdoors + outdoor wind + cold (< 40°F outdoors)
              + season trend + extra-point shift
              + accuracy[kicker] + leg[kicker] × (distance − 40) / 10
```

- **Distance curve:** a restricted cubic spline, with knots at 22, 33, 42, 49, 55 and
  62 yards. It is linear in the logit beyond 62 yards, so probability keeps falling past
  the longest kick ever attempted (70 yards). There is no cutoff.
- **Kicker effects:** each kicker gets an accuracy shift and a leg-strength slope, with
  Gaussian priors (sd 0.3 each, in logit units).
  - Kickers with few kicks are shrunk toward the league curve.
  - An unknown kicker gets the league curve.
  - The prior sds were tuned on 2023 and 2024 field goals, each fit only on earlier
    seasons.
- **Monotone:** every kicker's curve falls strictly with distance from 18 to 110 yards.
  This is tested.

## Validation (fit on 2014–2024, scored on 2025 field goals, out-of-sample)
| model | log loss | Brier |
|---|---|---|
| **kicker model** | **0.3600** | **0.1112** |
| same model, league curve only (no kicker effects) | 0.3646 | 0.1131 |
| nfl4th GAM (with its 58-yard decay and 0% from the 53) | 0.3650 | 0.1130 |

Make rate vs prediction by distance, 2025:

| distance | n | made | kicker model | nfl4th GAM |
|---|---|---|---|---|
| < 30 | 221 | 98.2% | 98.2% | 97.9% |
| 30–39 | 313 | 93.3% | 92.9% | 92.1% |
| 40–49 | 328 | 84.1% | 80.8% | 79.5% |
| 50–54 | 161 | 72.7% | 71.8% | 68.8% |
| 55–59 | 94 | 66.0% | 61.4% | 55.9% |
| 60+ | 23 | 52.2% | 35.3% | 29.8% |

All models under-predict 2025's long kicks. Two things likely contribute:
- **Selection:** teams try 55+ yard kicks only with good kickers in good conditions.
- **Kicking keeps improving:** the season trend carries only part of that.

The kicker model has the smallest error in every band.

**League curve** (outdoors, 2025): 40 yd 87%, 50 yd 74%, 55 yd 65%, 60 yd 47%, 65 yd 26%,
70 yd 12%, 75 yd 5%, 80 yd 2%.

**Active kickers** (indoors, 2025, at least 40 attempts), strongest and weakest from 60
yards:

| kicker | 50 yd | 60 yd | 65 yd | 70 yd |
|---|---|---|---|---|
| B. Aubrey | 85% | 70% | 51% | 32% |
| C. Boswell | 85% | 68% | 49% | 29% |
| C. Dicker | 84% | 65% | 44% | 25% |
| … | | | | |
| M. Badgley | 71% | 39% | 19% | 8% |

## Use in the decision engine
- **Distance:** kick distance = yardline_100 + 17. This is configurable as
  `DecisionConfig.kick_distance_offset`.
- **Kicker:** `kicker_player_id` (nflverse GSIS id) is an optional input column.
  - **Backtest:** the kicker who kicked for that team in that game, else the team's
    previous kicker.
  - **Live:** `nfl4th-py live` looks up the team's kicker from nflverse play-by-play
    for the current and previous season.
  - Without a kicker, the league curve is used.
- **Weather:** `wind` (mph) and `temp` (°F) are optional columns.
- **Hopeless attempts:** an attempt is offered only if the kicker's make probability is
  at least `DecisionConfig.fg_min_make_prob` (default 1%). Below that, there is no FG
  value, and the FG option can neither be recommended nor block a recommendation.
  - In the 2025 backtest, thresholds from 0.5% to 5% recommend the same field goals
    (856–858).
  - With no threshold, 575 recommendations are blocked by 80- to 100-yard "attempts".

## Effect on the 2025 backtest (VALIDATION.md)
- **Coverage:** 3,003 recommendations instead of 2,444; refusals fall from 38.9% to
  24.9%.
- **Calibration:** field goal EVs are unbiased against the settled market price
  (−0.08 points, 90% CI −0.29 to +0.13).
- **Kicker identity:** changes 2.1% of recommendations, mostly a strong kicker turning a
  44–50 yard go or punt into a field goal. On those plays coaches made the
  kicker-specific call 71% of the time.

## Limitations
- **No selection correction** for which kicks get attempted (see above).
- **Kicker skill is constant over a career,** apart from the league-wide season trend.
  There is no aging or form.
- **Wind and temperature are game-level values** from nflfastR, not conditions at the
  moment of the kick.
- **Blocked kicks are counted as misses,** and blocked-kick returns are not modeled.
