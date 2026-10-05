# DECISION — the market-valued 4th-down engine (Phase 6)

Entry point: `nfl4th_py.decision.add_4th_probs(df, market_snapshot, games=None, config=DecisionConfig())`.
Decisions can be logged with `decision.log_decisions(result, raw_quotes, directory)`.

## Inputs
- **`df`**: the nfl4th input columns (see PORTING_NOTES §1.1), plus optional `runoff`,
  `kicker_player_id`, `wind` and `temp`.
- **`market_snapshot`**: one row per play.

  | column | meaning |
  |---|---|
  | price | P(posteam wins) + ½·P(tie): the winner-market mid (live) or the pre-snap trade VWAP (historical) |
  | bid_ask | winner-market bid/ask width |
  | quote_age_s | seconds since the price was observed |
  | n_trades | trades (historical) or book levels (live) behind the price |
  | halted | market paused |
  | kalshi_spread_line, kalshi_total_line | Kalshi pregame implied home margin and total |
  | kalshi_spread_quality, kalshi_total_quality | ladder fit quality, 0–1 |

  The pregame betting lines in the schedule are **replaced** by the Kalshi values. If those
  are missing, the play is refused.

## Calculation
1. **Outcome probabilities** come from the ported nfl4th components:
   - the conversion model, with strength features from the Kalshi pregame spread/total;
   - the kicker-specific FG make model at every distance, with no cutoff (FIELD_GOALS.md;
     this replaces nfl4th's GAM, which is 0% from the 53 out);
   - the punt outcome table (no punt inside the 31).
2. **Outcome states**, from the team's view:

   | option | outcome | resulting state |
   |---|---|---|
   | go | converted | team ball at each possible spot |
   | go | TD | team at its **PAT**, +6 |
   | go | failed | opponent ball at the spot |
   | go | loss into own end zone | **safety**: −2, team free-kicks |
   | FG | make | +3, team **kicks off** |
   | FG | miss | opponent ball at the kick spot, no worse than its 20 |
   | punt | normal | opponent ball at the landing spot |
   | punt | block | opponent ball at the spot |
   | punt | muff | team recovers at the spot |
   | punt | return TD | opponent at its PAT, +6 |

   - **Clock:** the measured median game clock used until the next snap.

     | outcome | seconds |
     |---|---|
     | non-TD conversion | 38 |
     | non-TD conversion, last 2 min of Q2 | 9 |
     | non-TD conversion, last 2 min of Q4/OT, team leading | 40 |
     | non-TD conversion, last 2 min of Q4/OT, team tied or trailing | 12 |
     | failure, safety, TD | 5 |
     | FG make/miss | 4 |
     | punt | 9 |

     nfl4th uses 6 s for everything, which understated a conversion's clock use by about
     30 s and doubled a bias against going for it (VALIDATION.md §5).
     - A team that is not ahead stops the clock after a late conversion, so it always
       keeps at least 3 s.
     - The clock stops at the two-minute warning (Q2, Q4 and OT).
     - `runoff` overrides a conversion's clock use when 6 + runoff is larger. nfl4th
       ignores `runoff` (see PORTING_NOTES §5b).
   - **End of the 1st half:** the state becomes the Q3 kickoff to the 2nd-half receiver.
   - **Kneel-out states:** in Q4 or OT, the team with the ball (or receiving a kickoff)
     leads and the other side can't stop the clock (under 120/80/40 s with 0/1/2
     timeouts, nfl4th's rule). The leader's value is set to at least 0.99, and the
     trailing side's to at most 0.01.
     - 0.99 is the Kalshi median for 378 such 2025 states, which is the market's tick
       limit.
     - These states are rare in the training data, and the response model under-priced
       them. That made a late field goal while down 4–7 look worthwhile.
   - **End of game:** a decided outcome is valued exactly (1 or 0). A tie when time
     expires (overtime follows, or a regular-season tie that Kalshi settles at 0.50) is
     valued at 0.5 with a wide sd.
   - **Overtime:** supported, using OT's own clock. Plays carry a caution because the
     training data has few OT states.
3. **Valuation:** every state goes through the Phase 5 transition model, anchored on the
   live price. Each prediction comes with a price, a logit sd (calibrated) and a support
   percentile.
4. **Expected value:** EV(option) = Σ prob × predicted price.
   - **sd(option)** = Σ prob × sd(price). The delta method converts each logit sd to a
     price sd, and outcomes within an option are treated as fully correlated (they share
     one model and one game), which is conservative.
   - **Gap:** the best EV minus the second best.
   - **Gap sd** = √(sd_best² + sd_second²).
   - **`prob_best_option_is_best`** = Φ(gap / gap sd).
5. **Recommendation:** the option with the highest EV among the options the model can
   value, unless a hard rule fires. If one option's outcome states are outside the
   model's support, that option is left out and named in a caution, which also says
   whether its unreliable value would have ranked first. With fewer than two valuable
   options, the play is refused.

## Hard rules and cautions (`DecisionConfig`, all configurable)
A **hard rule** refuses the recommendation. A **caution** still recommends but sets
`confidence = "caution"` and lists the reason in `cautions`.

| condition | hard limit | caution | why |
|---|---|---|---|
| no winner price | refuse | — | nothing to anchor on |
| price near 0 or 1 | — (`min_price`/`max_price` off) | below 2% or above 98% | the market is at its 1¢ tick and barely moves; the model's price error there is ~0.2¢ |
| options within one tick at a tick-limit price | — | "toss-up" | the market can't separate the options |
| price age | > 120 s | 30–120 s | flips < 1.6% of recommendations even at 60 s (VALIDATION §3) |
| bid/ask width | > 10¢ | 4–10¢ | |
| no trade in 60 s | — | price from the last quote | |
| Kalshi pregame spread / total | missing | fit quality < 0.2, or a Kalshi-only fallback | weak ladders are still within 1–2 points of the betting line |
| overtime | `max_qtr` = 5 (set 4 to refuse) | always | few training states |
| last 2 min of Q4 | `q4_refuse_last_seconds` = 0 (off) | — | handled by the clock and kneel-out rules above |
| FG make probability < `fg_min_make_prob` | FG not offered | — | a hopeless attempt is not an option |
| option support (probability-weighted) < 0.01, or an outcome with prob ≥ 5% and support < 0.002 | refuse only if fewer than two options remain | option left out, named in a caution | outside the model's data |

The Kalshi-only pregame fallbacks are:
- **Spread:** if the ladder can't be fit (1 game in 2025), it is implied from the Kalshi
  pregame winner price with a normal margin (sd 13.5).
- **Total:** if the ladder can't be fit (5 games), the median Kalshi total of that week's
  games is used.

A refused play gets `recommendation = None`, `no_recommendation = True` and a
`no_recommendation_reason`. There is no WP-model fallback. `wp_baseline` is not imported
by this module, and `tests/test_quarantine.py` enforces that.

On all 3,997 4th-down plays in 2025 (out-of-sample fold models, current defaults):

| | plays |
|---|---|
| recommended | 3,827 (95.7%): punt 1,792, FG 1,087, go 948 |
| normal confidence | 3,197 |
| caution | 630 (335 are tick-limit toss-ups) |
| refused | 170 (4.3%), all model support except 4 very wide markets |

Breakdown in VALIDATION.md §2.

## Anchoring: does conditioning on the pre-play price bias counterfactual values?
The pre-snap price already reflects the market's expectation of the decision and its
outcome: P₀ ≈ Σ over options of P_market(option) × EV(option).

The transition model predicts each outcome's price as logit(P₀) + f(pre state, post
state, change).
- **Why anchoring is fine to first order.** P₀ is identical for all three options at a
  given snap, and it enters every outcome prediction the same way (as the base margin).
  It shifts all options together. The comparison between options is driven by f, which
  values the *states*: who has the ball where, the score and the clock. A state's value
  doesn't depend on which decision led to it, so the decision expectation inside P₀
  mostly cancels when options are compared.
- **What anchoring does buy.** P₀ carries information that isn't in the state features:
  team quality beyond the pregame line, injuries, weather, in-game form. Anchoring passes
  that information to every counterfactual state, which a static WP model can't do.
- **Remaining sources of bias** (not removed; to be checked in Phase 8):
  1. *The decision itself carries information.* The market may infer something from the
     choice (e.g. a team that punts on 4th & 1 is signaling low confidence). Training
     transitions come from real decisions, so f partly learns those signals for the
     decisions teams actually made. Counterfactual options inherit the signal of a
     *typical* team making that choice.
  2. *f is non-linear in P₀.* If the market expected a go and the team kicks, the post-FG
     price comes from f evaluated at the same P₀. Training contains few post-FG
     transitions from situations where the market expected a go, so these are
     extrapolations. The support measure partly flags this, since P₀ is one of its
     features.
  3. *Rare post-states.* Failed attempts, missed FGs and blocked punts have low support
     (median support percentile 0.03–0.04). Their values carry larger uncertainty, which
     the calibrated sd reflects.
- **Phase 8 check.** If the market is efficient and the engine is calibrated, then over
  actual 4th downs the average of EV(actual choice) − P₀ should be near 0, and the
  engine's predicted value of the *realized* outcome should match the settled price.

## Known limitation (also in README)
The conversion, FG and punt components don't adjust for in-game changes such as a backup
QB, an injured kicker or weather. The winner price reflects those changes, so outcome
**values** do too, but the outcome **probabilities** don't. Team strength is the Kalshi
*pregame* line, fixed for the game, by design.
