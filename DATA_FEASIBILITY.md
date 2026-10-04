# DATA_FEASIBILITY — aligning plays with Kalshi prices

## 1. Per-play wall-clock timestamps

### nflfastR (`nflreadr::load_pbp`) — usable, recommended primary source
Checked on 2024–2025 pbp (data updated 2026-08-13):

| column | meaning (evidence below) | coverage | precision |
|---|---|---|---|
| `time_of_day` | **snap time**, UTC ISO-8601 | 100% of rows in 2024 and 2025 | milliseconds (13% of rows are whole seconds) |
| `end_clock_time` | **end of play (whistle)**, UTC | 96% of run/pass rows; 75–79% of FG/punt/kneel rows; 5% of PATs; 85–87% overall | milliseconds |
| `start_time` | game kickoff, local time, minute precision | — | — |

Evidence that `time_of_day` is the snap and `end_clock_time` is the whistle: the median
`end_clock_time − time_of_day` is **0.8 s for kneels, 0.5 s for spikes, 4.5 s for runs,
5.1 s for passes, 3.7 s for FGs, and 8.6 s for punts** (2025). Consecutive snaps on
running-clock plays are separated by roughly the game-clock time elapsed (e.g. 15:00→14:27
is 32.3 s of wall clock).

Data quality (2025, 40,778 snaps): 12 snaps (0.03%) have a `time_of_day` earlier than the
previous `play_id`, and 2 have duplicate times. Most are sub-second inversions; one is a
6,828 s glitch (`2025_01_NYG_WAS` play 4541). I'll flag these with a monotonicity check
rather than trust them.

### ESPN `summary?event=` (what `get_4th_plays()` uses) — usable live
Each play has a `wallclock` (UTC, **whole seconds**). On `2025_20_BUF_DEN` (150 plays
matched on quarter and clock), `wallclock − nflfastR time_of_day` had a median of −0.2 s
and an IQR of [−0.45, +0.14] s. So ESPN `wallclock` is also the snap time, at 1 s
resolution. A few large outliers came from ambiguous clock matches, not from the feed.
ESPN has no end-of-play timestamp, so live end-of-play has to come from the next play's
snap or from the `modified` field (which is unreliable, since it's updated after the game).
For live decisions only the pre-snap price matters, and the latest quote at decision time
is enough.

### Not verified
- Whether `time_of_day` is the ball-snap instant or the time a GSIS operator entered it.
  The sub-second kneel durations suggest a true snap timestamp. Phase 4 will measure the
  market reaction lag empirically against it, which also tests this.
- Replay reversals: 221 2025 plays contain "REVERSED". nflfastR keeps a single row with the
  final ruling; the snap time is right, but the market may have moved on the original
  ruling and then reversed. Phase 4 will measure "settled" before the next snap, which
  includes the review, and flag these plays.

## 2. Volume of plays matchable to market prices

**Universe.** Kalshi game markets exist only from 2025-07-31 (KALSHI_NOTES §7).

| | games | snaps (run/pass/punt/FG/kneel/spike) | all plays incl. kickoffs/PATs |
|---|---|---|---|
| 2025 reg + post (nflfastR) | 285 | 38,216 | 42,603 |
| 2026 to date (wk 1–4) [estimate] | ~62 | ~8,300 | ~9,300 |
| **Total available now** | **~347** | **~46,500** | **~52,000** |
| Each future season | ~285 | ~38,000 | ~42,500 |

Preseason games have markets but no reliable nflfastR timestamps or comparable pricing. I'd
exclude them.

**Price sources, best to worst:**
1. **Own recorder** (orderbook midpoint, sub-second). This covers only games after it
   starts, so ~0 games today.
2. **Trade tape** (`/historical/trades`, `/markets/trades`). Prices are microsecond
   timestamped trades with taker side.
   - In-game winner markets average ~2.3 trades/s, with a max gap of ~19 s (KC–MIA 2026).
   - Snaps are ≥ 20 s apart (5th percentile of the snap-to-snap gap is 19 s).
   - So a last-trade/VWAP price in, e.g., the 5 s before the snap and a settled price
     before the next snap should exist for **nearly every winner-market snap**.
   - Trades bounce between bid and ask. Taker side lets us use the trade price as a
     bid/ask proxy, and estimate a microprice from short windows.
3. **1-minute candlesticks** with yes_bid/yes_ask OHLC. These give a true quote midpoint,
   but at 1-minute resolution, and the median snap gap is 43 s. Pre- and post-play prices
   would often fall in the **same candle**, which is useless for transitions. They're good
   for quality fields (spread width) and as a fallback for slow sequences such as timeouts,
   reviews, and quarter breaks.

**Estimate of matchable plays** (this is a projection, not a measurement; Phase 4 will
measure it):
- Winner market: ~90–97% of snaps have a trade-based pre-snap and pre-next-snap price,
  i.e. **~33–37k plays/season, ~42–45k available now**.
- Losses come from the 2-minute drill (snaps ~20–25 s apart, with the market reaction
  still in progress at the next snap) and blowouts (prices pinned near 0/1).
- Live spread/total ladder estimate at the snap: totals strikes trade ~50× less than
  winner markets. A trade-based ladder fit per play will be stale for many strikes. A
  rolling window (e.g. 60–120 s) plus 1-min candle quotes should produce a ladder fit for
  most plays, but with quote ages of tens of seconds. That's fine for a strength feature
  that changes slowly, and weak for anything faster.

## 3. Is it enough for rare outcome types?

Counts per regular+post season (nflfastR 2024 / 2025), and what's available in Kalshi-era
data now (≈ 1.2 seasons):

| outcome | per season | available now (≈) |
|---|---|---|
| 4th-down go attempts | 815 / 931 | ~1,100 |
| failed 4th downs (turnover on downs) | 350 / 417 | ~500 |
| 4th-down TDs | 74 / 100 | ~120 |
| FG attempts | ~1,050 | ~1,250 |
| FG 50+ attempts / misses | 278 / 87 | ~330 / ~100 |
| blocked FGs | 19 / 24 | ~28 |
| punts | ~2,100 | ~2,500 |
| blocked punts | 12 / 10 | ~12 |
| punt return TDs | 7 / 15 | ~15 |
| muffed punts lost | 24 / 15 | ~20 |
| backed-up punts (yl ≥ 90) | 73 / 94 | ~110 |
| INT + fumbles lost | 689 / 668 | ~800 |
| touchdowns | ~1,460 | ~1,750 |

**Assessment:**
- **Common transitions are enough.** Plays gaining or losing yards, first downs,
  touchdowns, made FGs, normal punts, and possession changes number in the thousands to
  tens of thousands.
- **Failed 4th downs (~500) and long FG misses (~100) are too few to learn on their own**,
  but they don't need to be. A failed 4th down is "possession change at the spot", the same
  post-state class as ~800 turnovers, and a FG miss is "possession change at spot + 8". The
  transition model must therefore be built on **state features** (who has the ball, where,
  score, clock), not play-type labels. That is what the brief's Phase 5 design does.
- **Blocked kicks (~40), return TDs (~15), and muffs (~20) cannot be learned or validated**
  in isolation. Their *post-states* are covered by other transitions: a blocked punt looks
  like a short-field possession change, and a return TD looks like any opponent TD. Their
  *probabilities* come from the nfl4th components. So the model can value them, but I can't
  validate those values against observed market reactions with any statistical power. I'll
  report them as "out-of-sample by construction".
- **End-of-half, end-of-game, and OT** states are sparse: ~285 games × a few plays each
  in the last 2 minutes of a close game, plus ~15 OT games/season. Expect wide uncertainty
  there. The support measure (Phase 5) should refuse predictions in poorly covered cells.
- **Validation power** for the decision backtest is limited to ~1,100 4th-down go attempts
  plus ~3,800 kicks/punts so far. Distinguishing two decision rules whose EV gap is ~1 WP
  point is not possible with confidence on that sample. VALIDATION.md will say so.
- Every week of recording adds ~2,500 plays. **Waiting until the end of the 2026 season
  roughly doubles the data.** This argues for starting the recorder immediately (Phase 2)
  and for a time-based holdout on late-2026 weeks.

## 4. Decisions (2026-10-04)

- **Price source: the Kalshi trade tape** (`/historical/trades`). Pre-snap and settled
  post-play prices are built from trades, with taker side used as a bid/ask proxy.
  1-minute candles are used only for quote-quality fields and as a fallback, and are
  tagged as such.
- **Scope: the 2025 season** (285 regular + postseason games with Kalshi winner, spread
  and total markets, ~38k snaps). 2026 games are out of scope for the training and
  validation data. The time-based holdout comes from the latest 2025 weeks (late regular
  season + playoffs).
- Order-book history is not available from Kalshi (KALSHI_NOTES §7). Depth fields in the
  dataset will be absent for 2025, and quality is measured from trade density, trade-price
  dispersion and candle bid/ask width instead.
