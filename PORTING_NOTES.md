# PORTING_NOTES — nfl4th (R) → nfl4th_py

Source examined: this repo at `aca8c66` (nfl4th 1.0.7), plus the installed R packages it
depends on at runtime: nflfastR 6.0.0, fastrmodels 2.1.0, xgboost (R) 3.2.1.1, R 4.6.1.
Every statement below was checked against the code or by loading the artifacts in R
(`R/sysdata.rda`, the release-hosted models). Line refs are to files in `R/`.

## 0. Discrepancies between the project brief and the code

| Brief says | Reality | Consequence |
|---|---|---|
| "FG make probability model (inherited from nflfastR)" | nfl4th ships its **own** `mgcv::bam` GAM (`fg_model` in `R/sysdata.rda`, trained in `data-raw/_punt_and_fg_models.R`). nflfastR's FG model (`fastrmodels::fg_model`) is not used by nfl4th. | Port the nfl4th GAM, not nflfastR's. |
| "the nflfastR spread-adjusted WP model values every post-play state" | Every post-play state is valued by the **average of two WP models**: (1) nflfastR's `wp_model_spread` (xgboost) and (2) nfl4th's own **home-team** WP xgboost model (`wp_model.rds`, originally `home_wp_model` from Ben Baldwin's `fourth_calculator`), which itself uses nflfastR's **EP model** as an input. (`apply_win_prob.R:46-109`) | `wp_baseline.py` needs 3 xgboost models (EP, wp_spread, nfl4th home WP), not 1. |
| Edge cases incl. **OT** in parity fixtures | nfl4th does not support OT. `prepare_df()` maps any `qtr` not in {2,4} to `quarter_seconds_remaining + 900`, so a `qtr = 5` play gets nonsense clock values; the nflfastR path filters `qtr <= 4` and `get_4th_plays()` also drops OT. | R has no meaningful OT output to match. I propose OT rows go into the fixtures **flagged as "R-unsupported"** and are excluded from the parity criterion (see §6). |
| Conversion model uses "pregame spread and total" | True, but via three derived features (`posteam_spread`, `total_line`, `posteam_total`) plus era and roof dummies. See §3.1 and the **live-substitution issue in §7**, which needs your decision. | — |
| xgboost models can be "saved in xgboost's JSON format" | True. All xgboost artifacts are UBJ raw vectors loadable by R xgboost 3.x; `xgb.save(b, "x.json")` works. **None of them carry feature names** (`variable.names()` is `NULL`), so feature order is purely positional (from the `select()` calls). | Python must build the matrix in the exact column order below; tests will pin it. |

## 1. Exported functions (NAMESPACE)

| Function | Inputs | Output |
|---|---|---|
| `add_4th_probs(df)` | data frame of 4th-down situations (see §1.1) | `df` + `go_boost, first_down_prob, wp_fail, wp_succeed, go_wp, fg_make_prob, miss_fg_wp, make_fg_wp, fg_wp, punt_wp` (rows that get filtered out get `NA`) |
| `add_2pt_probs(df)` | same columns as 4th downs minus `ydstogo`/`yardline_100` | `df` + `wp_0, wp_1, wp_2, conv_1pt, conv_2pt, wp_go1, wp_go2` |
| `get_4th_plays(gid)` | nflfastR `game_id` (e.g. `2020_20_TB_GB`) | 4th-down plays scraped from ESPN `site.api.espn.com/.../summary?event=<espn id>` with all `add_4th_probs()` inputs plus `game_id, espn_id, play_id, desc, yardline, home_score, away_score, type_text, index`. Derives timeouts by parsing "Timeout #n by XXX" and challenge text; sets a heuristic `runoff` (40 s with a 4Q lead and ≤5:00; run to the 2:00 warning when 2:07–2:46 remain). Drops OT, the last 15 s of Q4, timeouts, 2-minute warning and end-of-period rows. |
| `load_4th_pbp(seasons, fast = FALSE)` | seasons ≥ 2014 | nflfastR pbp + `add_4th_probs()` columns (or joined from the precomputed release file when `fast = TRUE`) + `go` (100/0/NA; NA for dead-ball penalties in run/pass formation) |
| `make_table_data(probs)` | one row from `add_4th_probs()` | 3-row table (Go / FG / Punt): `choice, choice_prob, success_prob, fail_wp, success_wp`, in percent |
| `make_2pt_table_data(probs)` | one row from `add_2pt_probs()` | 2-row table (Go for 2 / Kick XP), same columns |
| `nfl4th_clear_cache(type)` | `"games"`, `"fd_model"`, `"wp_model"`, `"all"` | deletes cached files in `tools::R_user_dir("nfl4th","cache")` |

### 1.1 Required input columns for `add_4th_probs()`

Two code paths, chosen by whether a `type` column exists (`wrapper.R:56-63`).

**Path A — user-built frame (`type` present).** Required:

| column | meaning |
|---|---|
| `home_team`, `away_team` | nflverse abbreviations (used to join the schedule) |
| `posteam` | team with the 4th down |
| `type` | `"reg"` or `"post"` |
| `season` | integer |
| `qtr` | 1–4 |
| `quarter_seconds_remaining` | seconds left in the quarter |
| `ydstogo` | yards to go |
| `yardline_100` | yards from the opponent's end zone |
| `score_differential` | posteam score − defteam score |
| `home_opening_kickoff` | 1 if the home team kicked off to start the game |
| `posteam_timeouts_remaining`, `defteam_timeouts_remaining` | 0–3 |
| `runoff` *(optional, default 0)* | extra seconds to run off after a successful non-TD conversion |

Everything else comes from the schedule join (`helpers.R:get_games_file`, keyed on
`home_team, away_team, type, season`): `spread_line`, `total_line`, `roof` → `era3/era4`,
`outdoors/dome/retractable`, `fg_model_roof`, `home_total`, `away_total`.
Note: the join key has no week, so it only works because home/away flips between division
meetings; a same-season repeat of the same home/away pair would duplicate rows.

**Path B — raw nflfastR pbp (`type` absent).** `prepare_nflfastr_data()` derives `type` from
`season_type` and filters `game_seconds_remaining > 15`, non-NA `half_seconds_remaining`,
`qtr`, `posteam`, `qtr <= 4`, then `down == 4`. Needs the Path A columns (minus `type`) plus
`season_type, game_seconds_remaining, half_seconds_remaining, down`.

In both paths `prepare_df()` **overwrites** `half_seconds_remaining` and
`game_seconds_remaining` from `qtr` + `quarter_seconds_remaining`, forces `down = 4`, and
adds `home_receive_2h_ko` (−1 / +1 / 0), `elapsed_share`, `spread_time`
(home-perspective), `posteam_spread`, `posteam_total`, `home_/away_timeouts_remaining`, and
`original_posteam`.

## 2. Models and data artifacts

| Artifact | Type | Location | Use |
|---|---|---|---|
| `fd_model` (4th-down yards gained) | xgboost `multi:softprob`, 76 classes, 11 features, 1124 rounds, base_score 0.5 | GitHub release `nflverse/nfl4th` tag `model_archive`, `fd_model.rds` (an R raw vector of UBJ) → cached in `R_user_dir` | conversion / gain distribution |
| `two_pt_model` | xgboost `binary:logistic`, 9 features, 20–21 rounds | `R/sysdata.rda` (UBJ raw) | 2-pt conversion (needed for TD valuation inside go-for-it) |
| `fg_model` | `mgcv::bam` binomial GAM, 10,694 kicks | `R/sysdata.rda` (also `data-raw/fg_model.Rdata`) | FG and XP make prob |
| `punt_df` | table, 5,483 rows: `yardline_100` (31–99), `yardline_after`, `pct`, `muff` | `R/sysdata.rda` (also `data-raw/punt_data.rds`) | punt outcome distribution |
| `wp_model` (nfl4th home WP) | xgboost `binary:logistic`, 11 features, 3775 rounds | release `model_archive/wp_model.rds` | **baseline only** |
| `fastrmodels::ep_model` | xgboost `multi:softprob`, 7 classes, 18 features, 525 rounds | R package `fastrmodels` 2.1.0 | **baseline only** (feeds `home_ep`) |
| `fastrmodels::wp_model_spread` | xgboost `binary:logistic`, 12 features, 534 rounds | R package `fastrmodels` | **baseline only** |
| schedules | table | `nflreadr::load_schedules()` (nflverse release CSV/parquet) | spread/total/roof/espn id |

## 3. Model features and feature engineering

### 3.1 Conversion model (`fd_model`) — features in order (`decision_functions.R:283-296`)

1. `down` (always 4 at prediction time; trained on 3rd and 4th downs)
2. `ydstogo`
3. `yardline_100`
4. `era3` = 1 if 2014 ≤ season ≤ 2017
5. `era4` = 1 if season ≥ 2018
6. `outdoors`
7. `retractable` (also used for `open`, `closed`, and missing roof)
8. `dome`
9. `posteam_spread` = `spread_line` if posteam is home else `-spread_line` (nflverse
   `spread_line` > 0 means the **home team is favored**, so this is posteam's expected margin)
10. `total_line`
11. `posteam_total` = `(total_line + posteam_spread) / 2`

Output: 76 probabilities for `yards_gained ∈ {-10, …, 65}` (class k ↔ k − 10). Training
truncated gains to [−10, 65] and coded first-down defensive penalties as a gain of
`max(ydstogo, penalty_yards)`. Trained on 2014–2019 regular-season 3rd/4th-down run, pass and
sack plays (`data-raw/_go_for_it_and_2pt_models.R`; the release model is `fd_model_v2` from
`fourth_calculator`, and its 1124 rounds match that script).

### 3.2 2-pt model — features in order (`decision_functions.R:183-194`)
`era2` (forced 0), `era3`, `era4`, `outdoors`, `retractable`, `dome`, `posteam_spread`,
`total_line`, `posteam_total`.

### 3.3 FG model
`sp ~ s(yardline_100, by = fg_model_roof) + fg_model_roof`, binomial, where
`fg_model_roof = paste0(roof == "outdoors", season >= 2020)` ∈ {`00`,`01`,`10`,`11`}.
The only inputs are an integer `yardline_100` and a 4-level factor. Post-processing in
`get_fg_wp()`:
- `yardline_100 > 40` (kick > 58 yd): `p = (53 − yl)/13 × p(yl = 40)` — linear decay to 0;
- `yardline_100 ≥ 53` (kick ≥ 71 yd): `p = 0`;
- XP make prob = `p(yl = 15)`.

### 3.4 Punt table
`P(yardline_after, muff | yardline_100)`, sums to 1 for each `yardline_100` in 31–99
(verified). `yardline_after` is from the **punting team's** view, so the receiver's
`yardline_100` is `100 − yardline_after`; `yardline_after == 100` is a return TD;
`yardline_after == yardline_100` (with `muff == 0`) is the block row; `muff == 1` means the
punting team recovers at `yardline_after`. **No rows for `yardline_100 ≤ 30`**, so `punt_wp`
is `NA` there and treated as 0 (punting is never recommended inside the 31).

Code observation, not something to change: in `data-raw/_punt_and_fg_models.R:96-112` the
bin probability assigned to the return-TD row is `bin_blocked_pct` and the block row gets
`bin_td_pct` (they look swapped). The shipped `punt_df` already contains that, and a
faithful port inherits it. The effect is tiny (both are ~0.1–0.5%).

### 3.5 nfl4th home WP model — features in order (`apply_win_prob.R:4-20`)
`home_receive_2h_ko, spread_time, home_posteam, half_seconds_remaining,
game_seconds_remaining, Diff_Time_Ratio, home_score_differential, home_ep, ydstogo,
home_yardline_100, home_timeouts_remaining` — **11** features. The `select()` lists
`home_timeouts_remaining` twice and dplyr de-duplicates, so **away timeouts never enter**.
The model metadata confirms `num_feature = 11`.

### 3.6 nflfastR `wp_model_spread` — features in order
`receive_2h_ko, spread_time, home, half_seconds_remaining, game_seconds_remaining,
Diff_Time_Ratio, score_differential, down, ydstogo, yardline_100,
posteam_timeouts_remaining, defteam_timeouts_remaining`, where nflfastR recomputes
`posteam_spread`, `elapsed_share`, `spread_time = posteam_spread·e^{-4·elapsed}`,
`Diff_Time_Ratio = score_differential / e^{-4·elapsed}` from the current row.

### 3.7 nflfastR `ep_model` — features in order
`half_seconds_remaining, yardline_100, home, retractable, dome, outdoors, ydstogo, era0,
era1, era2, era3, era4, down1, down2, down3, down4, posteam_timeouts_remaining,
defteam_timeouts_remaining`; `ep = 7·TD + 3·FG + 2·S − 7·oTD − 3·oFG − 2·oS`
(classes ordered TD, oTD, FG, oFG, S, oS, none). Eras recomputed from `season`.

## 4. The go / FG / punt calculation

All WPs are from the **original posteam's** perspective. "Valuing a state" below means
`calculate_win_probability()` in `apply_win_prob.R`, i.e. the mean of the two WP models.

### 4.1 Go for it (`get_go_wp`)
1. Predict 76 gain probabilities. Expand each play to 76 rows, gain −10…65.
2. Any `gain > yardline_100` is set to `yardline_100` (a TD) and those rows' probabilities
   are **summed into one TD row**.
3. For each row: `yardline_100 -= gain`; `turnover = gain < ydstogo`; on turnover flip
   possession, `yardline_100 = 100 − yardline_100`, negate the score. `down = 1`.
   If `yardline_100 == 0` add 6 to `score_differential`.
4. Run off 6 s. If converted and not a TD, also subtract `runoff`. Floor clocks at 0.
   `ydstogo = yardline_100 if < 10 else 10`.
5. TD rows: value = `max(wp_go1, wp_go2)` from `get_2pt_wp()`: probabilities from the XP
   (`fg_model` at yl 15) and 2-pt models, each outcome (0/1/2 points) valued as the
   opponent's 1st-and-10 at their 25 after a touchback kickoff.
6. Non-TD rows: `flip_half()` (if `qtr == 2` and the clock is now 0, move to the start of
   Q3 with the 2nd-half receiver at its 25), then value the state.
7. Kneel-out overrides (`decision_functions.R:399-433`): after a converted non-TD with the
   lead and `game_seconds < 120/80/40` with the defense holding 0/1/2 timeouts, WP = 1;
   after a failure with the opponent leading under the same clock and timeout conditions,
   WP = 0.
8. `first_down_prob = Σ prob(no turnover)`; `wp_succeed`/`wp_fail` are probability-weighted
   means within each group; `go_wp = Σ prob · wp`.

Edge cases a faithful port must reproduce:
- **Outcomes beyond the line to gain** are kept as distinct yardlines (every gain from
  `ydstogo` to `yardline_100 − 1`), each valued separately; beyond-the-goal-line mass is
  pooled into the TD.
- **Backed-up negative plays**: a loss that pushes past the posteam's own goal line gives
  `yardline_100 > 100`, which flips to ≤ 0 for the opponent. Exactly 0 is scored as an
  **opponent TD** (+6, routed through the 2-pt logic); negative values are passed to the
  WP models as-is. Reality would be a safety. These only matter at the 90+ yard line.
- Converted rows run off `runoff`; failed and TD rows only 6 s.
- `elapsed_share` (used in the home-WP model's `spread_time` and `Diff_Time_Ratio`) is
  computed **once** in `prepare_df()` and is **not** updated after the runoff or
  `flip_half()`. nflfastR's spread model recomputes it. The port must use the stale value
  for the home model.
- Timeouts fed to the EP/home-WP models are the **pre-play posteam's**, even after a
  possession flip (only the nflfastR model gets re-derived posteam/defteam timeouts).

### 4.2 Field goal (`get_fg_wp`)
- `fg_make_prob` from §3.3.
- Make: `flip_team()` (opponent ball, 1st & 10, −6 s), `yardline_100 = 75`, score −3
  from the opponent's view, `flip_half()`, value, `end_game_fn()`.
- Miss: opponent ball at `92 − yardline_100`, capped at 80 (a miss from inside the 20 is
  spotted at the 20), floored at 1. Same −6 s, `flip_half()`, value, `end_game_fn()`.
- **Blocked FGs and returns are not modeled.** A block is just a miss at the spot.
- `fg_wp = p·make + (1 − p)·miss`. It is never NA (p = 0 at the 53+).

### 4.3 Punt (`get_punt_wp`)
For each row of `punt_df` at this `yardline_100`: `flip_team()` (−6 s), receiver at
`100 − yardline_after`. If `muff == 1`, the punting team keeps the ball at
`yardline_after`. A return TD (`yardline_after == 100`) means the punting team receives a
kickoff at its 75, down by 7 more. `ydstogo = yardline_100 if < 10`. Then `flip_half()`,
value, `end_game_fn()`, and `punt_wp = Σ pct · wp`.
`end_game_fn()` sets WP = 0 when the team with the ball after the punt leads and can kneel
out the clock. On the muff and return-TD rows the "team with the ball" is the punting team,
so the override fires in the wrong direction (rare; port faithfully, flag in the code).

### 4.4 Recommendation
`go_boost = 100 · (go_wp − max(fg_wp, punt_wp, with NA punt → 0))`. There is no explicit
"recommendation" column. Callers read the sign of `go_boost`, and `make_table_data()`
orders the options.

## 5. What Python can load directly vs. what must be exported or refit

| Piece | Plan | Expected error |
|---|---|---|
| `fd_model`, `two_pt_model`, nfl4th `wp_model`, fastrmodels `ep_model`, `wp_model_spread` | `scripts/export_from_r.R`: `xgb.save(booster, "*.json")`; load with Python `xgboost.Booster(model_file=...)`; feed numpy arrays in the column order above. Store the expected order and `num_feature` in a sidecar JSON; a test asserts it. | float32 rounding, ~1e-7. |
| `fg_model` (GAM) | **Exact lookup table**, not interpolation: the inputs are an integer `yardline_100` and a 4-level factor, so I export `predict(fg_model, type = "response")` for yl 1–99 × 4 levels (396 values) to parquet. | 0 at integer yardlines (to float precision). Non-integer yardlines don't occur in either nflfastR or ESPN data. Refitting isn't needed. |
| `punt_df` | Export to parquet as-is. | exact |
| schedules (roof, era, espn id) | nflreadr's public release files (parquet/CSV) are readable directly from Python; pregame `spread_line`/`total_line` are used only by the baseline. | exact |
| nflfastR `calculate_expected_points` / `calculate_win_probability` logic | Re-implement the feature engineering (§3.5–3.7) in `wp_baseline.py`. | parity-tested |

Tooling found locally: R 4.6.1 with nfl4th 1.0.7, nflfastR 6.0.0, fastrmodels, xgboost
3.2.1.1, mgcv, nflreadr. The default `python` is 3.14 (miniconda), and `py -3.12` is also
available. Neither currently has `xgboost`, `pandas`, `polars` or `pyarrow`. Phase 1 will
add a `pyproject.toml` with `xgboost`, `numpy`, `pandas`, `pyarrow`, `requests`,
`websockets`, `cryptography` (Kalshi signing) and `pytest`.

## 6. Parity plan notes (for Phase 1)

- Fixture source: nflfastR 2023–2025 4th downs (≥ 2,000), stratified to include: the last
  2 minutes of Q2 (`flip_half` path), the last 2 minutes of Q4 with each timeout count
  (kneel overrides), goal-to-go, yl ≥ 90 (backed-up go and punt), yl 35–52 (long FG
  decay), yl ≥ 53 (FG prob 0), yl ≤ 30 (punt NA), plus a synthetic `runoff > 0` set.
- OT plays (`qtr == 5`): included for coverage but flagged `r_supported = False` and
  excluded from the 0.001 / 99.5% criteria, since R itself produces meaningless values.
- R on Windows: `add_4th_probs()` downloads `fd_model.rds` / `wp_model.rds` on first use
  (needs network). I will pin the downloaded files' SHA-256 in `data/MANIFEST.json`.

## 7. Issue needing your decision before Phase 3: what replaces "pregame spread/total"

The conversion and 2-pt models use the **pregame** expected margin and total as
team-strength features, measured from 0-0 over 60 minutes. Kalshi spread and total markets
settle on the **final** margin and total, so a live ladder mid-game prices
`current score + rest-of-game`. Plugging the live implied margin in directly would change
the feature's meaning, e.g. a team up 21 in Q4 would look like a 21-point favorite. That
breaks the brief's requirement that "the substitution preserves their meaning."

Options:
- **(A) Rest-of-game rescaling (my recommendation).**
  `strength_margin = (live_margin − current_margin) / f_remaining` and
  `strength_total = (live_total − current_points) / f_remaining`, where `f_remaining` is
  the fraction of expected scoring left (start with `game_seconds_remaining / 3600`; can be
  calibrated later from pbp scoring rates). This gives a full-game-equivalent spread and
  total, and it equals the pregame line at kickoff. It becomes noisy late in games, so I'd
  shrink toward the pregame-close implied values as `f_remaining → 0`.
- **(B)** Use the last Kalshi ladder before kickoff (a "Kalshi closing line") as the
  feature for the whole game. It's simple and has the same meaning as the training data,
  but it ignores in-game information. The market-response model would still see live
  ladders.
- **(C)** Raw live implied values. I recommend against this, because it changes the
  feature's meaning.
