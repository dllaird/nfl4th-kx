<!-- nfl4th-kx keeps this README by hand. README.Rmd documents the original R package. -->

# nfl4th-kx

nfl4th-kx is a fork of [nfl4th](https://github.com/nflverse/nfl4th), the R package behind
the [fourth down calculator](https://rbsdm.com/stats/fourth_calculator/). It adds
`nfl4th_py`, a Python package that recommends 4th-down decisions using
[Kalshi](https://kalshi.com) prediction-market prices instead of a win-probability model.

nfl4th compares the win probability of going for it, kicking a field goal and punting,
and values every possible outcome with a win-probability model. nfl4th-kx keeps
nfl4th's outcome probabilities and values each outcome with a model of how the Kalshi
game-winner market reprices after plays:

- **Outcome probabilities** come from nfl4th's components, ported to Python and checked
  against R: the conversion model and the punt outcome table. Field goals use a
  kicker-specific make-probability model with no distance cutoff. Team strength comes from the Kalshi spread and total markets just before
  kickoff, instead of the betting line.
- **Outcome values** come from a market-response model. It predicts the winner-market
  price once the market has settled after a play, from the price before the snap and the
  change in game state. It is trained on every play of the 2025 season, aligned to
  Kalshi trades.
- **The decision** is the option with the highest expected post-play price. It comes
  with the gap to the next option and its uncertainty. When the market or the model
  isn't reliable enough, the engine gives no recommendation and says why. It never falls
  back to a win-probability model.

## Status

| Part | State |
|---|---|
| Python port of nfl4th | Matches R on 2,180 fourth downs: probabilities and win probabilities within 2e-7, the same recommendation on every play |
| Kalshi data | All 285 games of the 2025 season: winner, spread and total markets. Trade tape and 1-minute candles, via the public API |
| Market-response model | Average error of 1.1 cents on post-play prices, vs 1.5 for nfl4th's win-probability models anchored to the same price (out-of-sample) |
| Field goals | Kicker-specific model, out-of-sample on 2025 field goals: log loss 0.360 vs 0.365 for nfl4th's model |
| Decision engine | Out-of-sample backtest on 3,997 2025 fourth downs. Recommends on 95.7% (3,827), including the final 2 minutes and overtime; 583 carry a caution |
| Live mode | `nfl4th-py live`, tested against live Kalshi orderbooks but not yet during a game |

Details:

| Document | What it covers |
|---|---|
| [PORTING_NOTES.md](PORTING_NOTES.md) | How nfl4th works and how it was ported, including R quirks reproduced for parity |
| [KALSHI_NOTES.md](KALSHI_NOTES.md) | The Kalshi API, NFL market structure and settlement rules |
| [DATA_FEASIBILITY.md](DATA_FEASIBILITY.md) | Play timestamps, data volume and scope decisions |
| [DATASET.md](DATASET.md), [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | The plays-to-prices training dataset |
| [RESPONSE_MODEL.md](RESPONSE_MODEL.md) | The market-response model and how it was validated |
| [DECISION.md](DECISION.md) | The decision engine, its refusal rules and anchoring |
| [FIELD_GOALS.md](FIELD_GOALS.md) | The kicker-specific field goal model |
| [VALIDATION.md](VALIDATION.md) | Backtest, sensitivity, market usability and caveats |

## Installation

Python 3.11 or later:

``` sh
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # on macOS/Linux: .venv/bin/python
```

All Kalshi access is read-only and uses public market-data endpoints, so no API key is
needed. The package contains no order-placement code.

## Usage

Run the engine on fourth downs. `df` has the same columns as nfl4th's `add_4th_probs()`,
and `snapshot` has one row of market data per play:

``` python
import pandas as pd
from nfl4th_py import decision

play = pd.DataFrame([dict(
    home_team="NO", away_team="ATL", posteam="NO", type="reg", season=2026,
    qtr=3, quarter_seconds_remaining=420, ydstogo=2, yardline_100=38,
    score_differential=-3, home_opening_kickoff=1,
    posteam_timeouts_remaining=2, defteam_timeouts_remaining=3,
)])
snapshot = pd.DataFrame([dict(
    price=0.45, bid_ask=0.01, quote_age_s=2, n_trades=10, halted=False,   # winner market, posteam view
    kalshi_spread_line=0.9, kalshi_total_line=47.8,                       # Kalshi pregame lines
    kalshi_spread_quality=0.6, kalshi_total_quality=0.9,
)])
out = decision.add_4th_probs(play, snapshot)
out[["go_ev", "fg_ev", "punt_ev", "recommendation", "ev_gap", "no_recommendation_reason"]]
```

Follow a live game. It prints a decision table at each 4th down and logs every decision
to parquet:

``` sh
nfl4th-py live --game 2026_04_ATL_NO
```

Run the original nfl4th calculation in Python (benchmark only; the decision engine never
uses it):

``` python
from nfl4th_py import prep
from nfl4th_py.models.wp_baseline import add_4th_probs_baseline
add_4th_probs_baseline(play, prep.load_games())
```

## Rebuilding the data and models

The Kalshi data and the dataset built from it live in `data_cache/`, which is
git-ignored. The scripts recreate them in this order:

| Step | Script |
|---|---|
| Export nfl4th and nflfastR models from R | `scripts/export_from_r.R` |
| Export 2014–2025 kicks and fit the kicker model | `scripts/export_kicks.R`, `scripts/train_kicker.py` |
| Build R parity fixtures | `scripts/make_fixtures.R` |
| Download 2025 Kalshi trades and candles (about 1.5 hours) | `scripts/backfill_kalshi.py` |
| Align plays to prices | `scripts/build_dataset.py` |
| Measure market reaction lag | `scripts/reaction_lag.py` |
| Train and validate the response model | `scripts/train_response.py`, `scripts/analyze_response.py` |
| Run the decision backtest | `scripts/backtest.py` |

Tests: `python -m pytest`.

## Known limitations

- **One season of data.** Kalshi NFL game markets start in 2025. Most 4th-down
  comparisons are not statistically significant on the late-season holdout.
- **Conversions are undervalued by about half a point** (+0.63, 90% CI +0.06 to +1.25). The response model undervalues
  successful 4th-down conversions, so the engine leans slightly against going for it.
- **Some plays get no recommendation, and some carry a caution.**
  - The engine refuses about 4% of plays, mostly when fewer than two options can be
    valued within the model's training data.
  - It flags recommendations made with marginal inputs: prices at the 1¢ tick limit
    (where the market can't separate options), stale or wide markets, weak pregame
    ladders, or overtime.
- **No in-game adjustments to outcome probabilities.** The conversion and punt
  components don't account for changes such as a backup QB. The field goal model knows
  the kicker and the game's weather, but picks up a replacement kicker only after his
  first kick of the game. Kalshi prices reflect those changes, so outcome *values* do too, but outcome
  *probabilities* don't. Team strength is the Kalshi pregame line, fixed for the game.

## The original R package

The R package is unchanged and still lives here (`R/`, `tests/testthat/`). For nfl4th
itself, see [nfl4th.com](https://www.nfl4th.com/) and the
[upstream repository](https://github.com/nflverse/nfl4th):

``` r
install.packages("nfl4th")
```

## Credits and license

nfl4th is by Ben Baldwin, with contributions from Sebastian Carl and the nflverse
contributors. nfl4th-kx also uses models and data from
[nflfastR](https://www.nflfastr.com/), fastrmodels and
[nflverse](https://github.com/nflverse). All are used under the MIT license, which this
fork keeps (see [LICENSE.md](LICENSE.md)).

Kalshi market data is read from Kalshi's public API and is not redistributed in this
repository.
