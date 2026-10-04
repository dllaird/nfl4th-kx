# KALSHI_NOTES — Kalshi Trade API v2 for NFL market data

Checked 2026-10-04 against docs.kalshi.com (`openapi.yaml` v3.32.0, `asyncapi.yaml`, the
getting_started pages) and with unauthenticated GET calls to production. No authenticated
or write calls were made. **[V]** = verified from the docs or a live response;
**[I]** = inferred.

## Discrepancies vs. the brief
- **Kalshi has no orderbook history**, by REST or otherwise [V]. Candlesticks give
  top-of-book bid/ask OHLC per minute, so the brief's assumption is correct, and the
  recorder is the only way to build depth history.
- **The WebSocket requires an API key**, even for public channels [V]. The recorder can
  run unauthenticated by polling REST (see §5), but the WebSocket path needs you to create
  a read-only key. Please confirm you're OK with that.
- **Prices are dollar strings, not integer cents.** The legacy cent and count fields have
  been removed from responses [V]. Contracts can be fractional (min 0.01).
- **Only ~1.2 seasons of NFL game markets exist** (from 2025-07-31). See §6 and
  DATA_FEASIBILITY.md.
- **The Super Bowl winner isn't in the game-winner series** (it's in `KXSB`), and the
  2025 AFC Championship is missing from `KXNFLGAME`.

## 1. Endpoints [V]
| | REST | WebSocket |
|---|---|---|
| Prod (recommended) | `https://external-api.kalshi.com/trade-api/v2` | `wss://external-api-ws.kalshi.com/trade-api/ws/v2` |
| Prod (legacy, works) | `https://api.elections.kalshi.com/trade-api/v2` | `wss://api.elections.kalshi.com/trade-api/ws/v2` |
| Demo | `https://external-api.demo.kalshi.co/trade-api/v2` | `wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2` |

## 2. Auth [V]
- Headers: `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-TIMESTAMP` (ms), and
  `KALSHI-ACCESS-SIGNATURE`. The signature is base64 of an RSA-PSS (SHA-256, MGF1-SHA256,
  salt = digest length) signature over `timestamp + METHOD + path`, where the path excludes
  the query string. Ed25519 keys are also accepted.
- **Market data REST is public.** Series, events, markets, orderbook(s), trades,
  candlesticks, batch candlesticks, all `/historical/*` endpoints, and exchange
  status/schedule all returned 200 with no auth.
- **The WebSocket connection always needs auth.**
- Plan: credentials come from `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH` env vars.
  `.env` and `*.pem` go in `.gitignore`. The client will contain no order endpoints at all.

## 3. Rate limits [V]
- Token buckets: most requests cost 10 tokens, and reads and writes are budgeted
  separately.
- Basic tier: 200 read tokens/s, about 20 req/s, with bursts up to 3 s of budget.
- A throttled request gets HTTP 429 `{"error":"too many requests"}`, with **no**
  `Retry-After` or rate-limit headers. Use exponential backoff with jitter.
- Unauthenticated limits are undocumented. About 3 req/s sustained produced no 429s.
  I'll default the client to ≤ 10 req/s and make that configurable.

## 4. Prices, units, orderbook [V]
- Prices are `*_dollars` strings with up to 4 decimals (`"0.5800"`). Sizes are `*_fp`
  strings with 2 decimals.
- NFL markets use `price_level_structure: "linear_cent"` (tick $0.01). Always read
  `price_ranges` rather than assuming the tick.
- **Historical endpoints use different keys** (`price.close`, `volume`, `yes_bid.close`,
  without the `_dollars`/`_fp` suffixes). The client must normalize both shapes.
- `GET /markets/{ticker}/orderbook?depth=0..100` returns **bids only**:
  `orderbook_fp.yes_dollars` and `orderbook_fp.no_dollars` as `[price, size]`, sorted
  ascending, so the best bid is the last element.
  - **Asks must be derived:** YES ask = 1 − best NO bid; NO ask = 1 − best YES bid.
  - Example: YES bids …0.59, NO bids …0.40 → YES 0.59 / 0.60.
  - A one-sided book has no ask on the other side. Midpoint is undefined then, and must be
    flagged rather than imputed.
- `GET /markets/orderbooks?tickers=…` fetches up to 100 books per call.
- WebSocket channels:
  - `orderbook_delta`: an `orderbook_snapshot` followed by deltas carrying
    `price_dollars, delta_fp, side, ts_ms`, with a per-subscription `seq` for gap
    detection.
  - Also available: `ticker`, `trade`, and `market_lifecycle_v2` (carries `is_deactivated`
    for pauses).

## 5. NFL market structure and discovery [V]
| Series | Market(s) per game | Strike |
|---|---|---|
| `KXNFLGAME` | **one binary per team** (`…-KC`, `…-MIA`); event `mutually_exclusive` | `strike_type: structured`, `custom_strike.football_team` (uuid) |
| `KXNFLSPREAD` | **one ladder per team**: "KC wins by over 13.5 points" | `strike_type: greater`, `floor_strike` = X.5, `custom_strike.football_team` |
| `KXNFLTOTAL` | one ladder: "Over 45.5 points" | `strike_type: greater`, `floor_strike` |

- Event ticker: `{SERIES}-{YY}{MON}{DD}{AWAY}{HOME}`, e.g. `KXNFLGAME-26SEP27KCMIA`.
  `sub_title` looks like "KC vs MIA (Sep 27)".
- **The ticker date can differ across series for the same game** (a flexed game:
  `KXNFLGAME-26JAN04SEASF` vs `KXNFLSPREAD-26JAN03SEASF`). Discovery must match on team
  codes plus a date within ±1 day, never on a constructed ticker.
- Kalshi team codes need a mapping to nflverse codes (e.g. `LA`/`LAR`, `WAS`/`WSH`,
  `JAC`/`JAX`). I'll build that from `yes_sub_title` and `custom_strike`.
- Discovery flow: `GET /events?series_ticker=…&status=open` (paginate by cursor), then
  `GET /events/{t}?with_nested_markets=true`. For settled pre-cutoff games, use
  `/historical/markets?event_ticker=…`.
- Ladder size: 2026 games run 26–31 spread strikes (both teams combined) and 19–22 total
  strikes. In 2025 the medians were 16 and 10.
- Ladders are asymmetric: the favorite has more strikes.
- **Winner normalization:** P(posteam wins) comes from posteam's market. As a consistency
  check, use 1 − the other team's market; the two should sum to about 1 (plus spread/vig).
  Ties settle at 0.50 for both (§7), so `P(A) + P(B) = 1` holds in expectation, and the
  price is E[payout] = P(win) + 0.5·P(tie).

## 6. Resolution: ties and overtime [V]
- **Winner:** a tie settles **$0.50 per team** (`rules_secondary`). Verified on
  `KXNFLGAME-25SEP28GBDAL-GB` (GB–DAL tie): `settlement_value_dollars = 0.5000`.
  - A game that starts within 48 h of schedule settles on the result; otherwise it
    settles at a fair price.
- **Spread:** OT counts toward the margin. On a tie, every "wins by over X.5" resolves No.
- **Total:** the full game including OT.
- **Abandoned games:** spread and total settle on stats if abandoned after 55 minutes;
  otherwise at the "last fair market price".
- Settlement source: the NFL (nfl.com).
- Markets close early once the outcome is determined (e.g. winner close 20:00:06Z,
  settled 20:02Z).
- Implication for the decision engine: the market value is **P(win) + ½·P(tie)**, not
  P(win). This is a small difference except in OT.

## 7. Historical data [V]
- **Live/historical split:** `GET /historical/cutoff` currently returns
  `market_settled_ts` = `trades_created_ts` = **2026-08-05**.
  - Markets settled before the cutoff, and trades before it, are **only** on
    `/historical/markets`, `/historical/markets/{t}/candlesticks`, and `/historical/trades`.
  - **The entire 2025 season is historical-only.**
  - Events and series stay on the live endpoints.
- **Trades:** `GET /markets/trades` (and `/historical/trades`) with `ticker`, `min_ts`,
  `max_ts`, `limit` ≤ 1000, and a cursor. Fields: `created_time` (μs), `count_fp`,
  `yes_price_dollars`, `taker_side`, `is_block_trade`. Trades are available for every NFL
  market back to the first game.
- **Candlesticks:** `period_interval` ∈ {**1, 60, 1440 minutes**}, so 1-minute is the
  finest. Max 5,000 candles per request; batch endpoint takes ≤ 100 tickers and ≤ 10,000
  candles. Each candle has price OHLC/mean plus **yes_bid and yes_ask OHLC**, volume, and
  OI.
- **Orderbook history:** none.

### Coverage (events, by ticker date) [V]
| Series | Earliest game | 2025 pre / reg / post | 2026 to date |
|---|---|---|---|
| KXNFLGAME | 2025-07-31 (HOF game) | 49 / **272** / 11 (SB in `KXSB`, AFC CG missing) | wk 1–4 played, listed through Oct 12 |
| KXNFLSPREAD | 2025-08-21 | 16 / **272** / 13 | same |
| KXNFLTOTAL | 2025-08-21 | 16 / **272** / 13 | same |

- No per-game NFL winner markets exist before July 2025.
- Trades, candles, and spread/total history all start with those first events. There is
  no orderbook history at all.
- Usable regular season plus playoffs: **≈ 285 games in 2025 + ≈ 60–65 games played in 2026 (weeks 1–4) [I]**
  so far.

### Liquidity [V]
| Game (2026) | Winner (2 mkts) vol | Spread ladder vol | Total ladder vol |
|---|---|---|---|
| KC–MIA (Sun) | 12.3M | 5.6M (26 strikes; top 1.9M) | 1.1M (19; top 0.16M) |
| LAC–BUF (Sun) | 15.2M | 4.8M | 1.4M |
| PHI–CHI (MNF) | 75.8M | 25.9M | 10.6M |
| PIT–NE (Sun) | 5.6M | 1.9M | 0.6M |

- Trade counts for KC–MIA: winner 108,898, spread 32,706, total 8,019.
- Median per 2025 event: winner 10.7M, spread 1.45M, total 0.72M.
- **Winner ≈ 2–5× the whole spread ladder and 7–15× the whole totals ladder.** Strikes
  away from the line are thin (5k–50k contracts).
- In-game trading in the winner market is continuous. KC–MIA had 28,785 trades during the
  game (~2.3/s), and the longest gap was 18.6 s.

## 8. Status, halts [V unless noted]
- Market `status` values: initialized, inactive, active, closed, determined, disputed,
  amended, finalized. The `status` *query filter* uses a different vocabulary: unopened,
  open, paused, closed, settled.
- Pauses appear as `is_deactivated` on `market_lifecycle_v2`. That REST `inactive` is the
  paused state is [I].
- There's a weekly exchange maintenance window, Thursday 03:00–05:00 (timezone [I]: ET).
  TNF games end before it.
- **Not verified:** whether Kalshi pauses NFL markets during reviews or injuries. The
  recorder will log lifecycle events to answer this.

## 9. Recorder implications
- Without an API key: poll `GET /markets/orderbooks` (100 tickers/call). A full Sunday
  slate is ~9 games × ~50 markets ≈ 450 markets ≈ 5 calls per poll. A 1 s poll stays within
  the documented Basic budget, but the unauthenticated limit is unknown, so it will back
  off adaptively.
- With a key (preferred): WebSocket `orderbook_delta` + `trade` + `ticker` +
  `market_lifecycle_v2`, plus a periodic REST snapshot to verify the book.
- In both cases: discover markets daily, and write partitioned parquet
  (`series/date/event`).

## 10. Findings from the 2025 backfill (2026-10-04)
- **Unauthenticated rate limit:** 8 req/s (4 threads) drew HTTP 429s; 3–4 req/s drew only
  occasional ones (14 retries over ~25k requests, all recovered). The full backfill took
  ~95 minutes and is 650 MB.
- **Ticker oddities:**
  - Two games use city names instead of codes (`…-BUFFALO`, `…-PHILADELPHIA`).
  - Kalshi uses `LA` and `LAR` interchangeably for the Rams.
  - One divisional game has a second, never-traded `KXNFLGAME` event dated a day earlier.
    Discovery keeps the highest-volume event per game and series.
- **Postseason winner markets:**
  - AFC Championship: there is no `KXNFLGAME` event.
  - NFC Championship: the `KXNFLGAME` event exists with **zero volume**.
  - Super Bowl: there is no `KXNFLGAME` event.
  
  Trading for all three happened in `KXNFLAFCCHAMP`, `KXNFLNFCCHAMP` and `KXSB`.
  `discovery.games_needing_champion_fallback()` handles this.
- **Ladders are much thinner than winner markets:**
  - In week 1, most spread strikes showed placeholder quotes at kickoff (a 0 bid, or a
    20–40¢-wide market) while still trading. Strike prices therefore fall back to the
    last hour's volume-weighted trade price.
  - From week 2 on, every game had a usable spread and total ladder.
- **Ladders open early:** strikes are added during the game. Only strikes open before
  kickoff are used for the pregame line.
