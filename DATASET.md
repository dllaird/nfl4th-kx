# DATASET — 2025 plays aligned to Kalshi winner prices (Phase 4)

Build: `scripts/backfill_kalshi.py`, then `scripts/build_dataset.py`. Timing study:
`scripts/reaction_lag.py`.

Outputs (in `data_cache/`, which is git-ignored because it is derived Kalshi data):
- `data_cache/dataset/plays_2025.parquet` (45,186 rows, 12.8 MB)
- `data_cache/kalshi_pregame_2025.parquet` (one row per game)

Columns are described in `DATA_DICTIONARY.md`.

## Scope and sources
- **Games:** all 285 2025 regular-season and postseason games. Winner prices come from
  `KXNFLGAME`, except for three games that use the championship series:
  - **AFC Championship:** `KXNFLAFCCHAMP`; there is no `KXNFLGAME` event.
  - **NFC Championship:** `KXNFLNFCCHAMP`; its `KXNFLGAME` event exists but never traded.
  - **Super Bowl:** `KXSB`.
  
  During those games, each championship market settles on that game alone, so it works as
  a winner market.
- **Plays:** nflfastR 2025 run, pass, punt, field goal, kickoff, extra point, kneel, spike
  and penalty no-plays. Timeout rows are excluded: nflfastR logs them as no-plays with
  out-of-order timestamps.
- **Prices:** the Kalshi trade tape for the two winner markets, merged into one home-win
  series. A home-market YES trade at x counts as x; an away-market trade at x counts as
  1 − x. Prices are then converted to the **pre-play posteam's** view.
  - Every price means **P(win) + ½·P(tie)**, because ties settle at $0.50.
  - 1-minute candles provide quote quality fields only. Kalshi has no orderbook history,
    so depth is not available for 2025.
- **Team strength:** the Kalshi pregame implied spread and total at kickoff (Phase 3),
  held fixed for the game. Six week-1 games have no usable spread or total ladder; their
  strength fields are NaN. There is no fallback to the betting line.

## Timing definitions (measured, not assumed)
- **Snap** = nflfastR `time_of_day`. The median `end_clock_time − time_of_day` is 0.8 s for
  kneels, 0.5 s for spikes and 4.5 s for runs, so `time_of_day` is the snap and
  `end_clock_time` is the whistle.
- **Reaction lag** (`scripts/reaction_lag.py`; 8,282 plays with a precise end time and a
  move of at least 3¢). Median share of the eventual move completed, by seconds after the
  whistle:

  | seconds after whistle | 0 | 2 | 4 | 6 | 8 | 10 | 15 | 20 | 25 | 30 | 40 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | share of move done | 0.07 | 0.22 | 0.39 | 0.57 | 0.68 | 0.76 | 0.87 | 0.91 | 0.94 | 0.96 | 0.97 |

  Measured from the snap instead, **the share of the move done is 0.00 at every offset
  from −10 s to 0 s** for both precise and whole-second timestamps. So there is no
  pre-snap leakage, and the coarse timestamps are not systematically late.
- **Pre-snap price** = VWAP of trades in the 5 s ending at the snap. If that window is
  empty, the last trade before the snap, provided it is at most 60 s old.
- **Settled price** = VWAP of trades in the 5 s ending 1 s before the **next** snap. The
  window never starts before the whistle (or before the snap, if the whistle time is
  missing). If the window is empty, the last trade after the whistle is used, provided it
  is at most 60 s old.
- **Unsettled:** next snap less than 25 s after the whistle (≈94% of the reaction done).
  These plays are flagged and excluded from `usable`. They are mostly hurry-up and
  2-minute-drill plays (5.0% of plays).

## Stoppages, reviews, penalties
- **Reviews and replay reversals:** the settled price is taken before the next snap, so it
  already reflects the final ruling, including any reversal. These plays are flagged
  `flag_replay` (331 usable) and `flag_reversed` (217 usable). nflfastR keeps one row with
  the final ruling. The market's move on the original call and its reversal are not
  modeled separately.
- **Long stoppages** (injuries, TV breaks, quarter breaks): the settled window is still
  just before the next snap. `settle_time_s` records the gap (p90 177 s), and
  `flag_qtr_change` marks quarter and half breaks.
- **Penalties:** penalty no-plays are kept as rows. Their outcome is a yardage change with
  no down change. `flag_penalty` and `flag_no_play` mark them.
- **Halts:** Kalshi exposes no historical halt or pause status. A market with no trade in
  the 60 s before the snap is treated as halted or stale (`flag_pre_stale`, 0.65% of
  plays), and those plays are excluded from `usable`.

## Alignment flags and the `usable` rule
`usable` requires all of the following:
- a pre-snap price and a settled price both exist;
- the pre-snap price is not stale;
- the timestamps are monotonic (0.06% of plays fail this; one game has a 6,800 s glitch);
- the play is not unsettled.

| flag | share of plays |
|---|---|
| flag_coarse_time (whole-second snap; 99% of PATs, ~25% of punts/FGs) | 10.4% |
| flag_missing_end (no `end_clock_time`) | 10.8% |
| flag_unsettled | 5.0% |
| flag_post_stale | 1.9% |
| flag_qtr_change | 1.9% |
| flag_pre_stale | 0.65% |
| flag_nonmonotonic_time | 0.06% |

Coarse-time plays are kept as usable, since the event study shows no leakage. The flag
stays so Phase 5 can evaluate them separately.

## Result
- **45,186 plays, of which 42,036 (93.0%) are usable;** 41,183 usable plays also have
  both Kalshi pregame lines.
- Usable rate by play type: run 93.5%, pass 91.0%, punt 98.6%, FG 96.3%, kickoff 98.6%,
  PAT 99.1%, kneel 52% (the market has often closed by end-of-game kneels).
- **4th downs:**

  | 4th-down outcome | total | usable |
  |---|---|---|
  | go-for-it attempts | 931 | 841 |
  | converted | 514 | 443 |
  | failed | 421 | 401 |
  | FG attempts | 1,024 | 1,008 |
  | FG missed or blocked | 140 | 139 |
  | FG blocked | 23 | 23 |
  | punts | 2,042 | 2,013 |
  | punts blocked | 10 | 10 |

- **Price quality** (usable plays):
  - median of 10 trades in the 5 s pre-snap window, with the newest trade 0.33 s old;
  - candle bid/ask spread: median 1¢, p90 2¢;
  - trade VWAP vs the candle midpoint (candle < 60 s old): median |diff| 0.5¢, p90 2.6¢,
    correlation 0.998;
- **Price source** (all plays): VWAP for 92.6% of pre-snap and 92.0% of settled prices;
  last trade for 6.8% and 6.1%.
- **Extreme prices:** 11.6% of usable plays have a pre-snap price below 0.03 or above
  0.97. Phase 5 must handle or refuse these.
- **Typical move:** |settled − pre-snap| has a median of 1.2¢ and a p90 of 5¢.

## Kalshi pregame strength (Phase 3) vs the betting line

| | usable games | corr with betting line | mean diff | MAE | p90 abs diff | max abs diff | median fitted sd |
|---|---|---|---|---|---|---|---|
| spread | 284/285 | 0.994 | +0.09 | 0.53 | 1.11 | 2.82 | 11.9 |
| total | 280/285 | 0.991 | +0.08 | 0.44 | 0.89 | 2.38 | 12.8 |

This confirms the substitution preserves the features' meaning: expected final home
margin and expected total, from 0-0 over a full game.
- **Ladder fits:** the normal fit was chosen for all games but one; the skew-normal rarely
  beats it by enough to be used.
- **Winner market vs ladder:** median |winner price − ladder-implied P(home win)| is 1.4¢,
  p90 3.5¢; 11 games exceed 5¢.
- **Missing ladders:** six week-1 games (`2025_01_ARI_NO`, `_CAR_JAX`, `_LV_NE`, `_MIA_IND`,
  `_NYG_WAS`, `_TB_ATL`) lack a usable spread or total ladder. They get no
  recommendation.

## Caveats
- **Only one season, 285 games.** Rare transitions (blocked kicks, return TDs) have tens
  of examples at most.
- **Prices are trade-based,** with no orderbook depth. Late in blowouts, trading thins and
  prices pin near 0/1.
- **The `time_of_day` and `end_clock_time` semantics are inferred,** from play durations
  and from market behavior. The data has no documentation of them.
- **PATs** almost all have coarse timestamps and no end time. Their windows are anchored
  on the snap.
