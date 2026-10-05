# DECISION — the market-valued 4th-down engine (Phase 6)

Entry point: `nfl4th_py.decision.add_4th_probs(df, market_snapshot, games=None, config=DecisionConfig())`.
Decisions can be logged with `decision.log_decisions(result, raw_quotes, directory)`.

## Inputs
- **`df`**: the nfl4th input columns (see PORTING_NOTES §1.1), plus optional `runoff`.
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
   - the FG make model (with nfl4th's decay beyond 58 yards);
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

   - **Clock:** 6 s per play. `runoff` is subtracted after a non-TD conversion; nfl4th
     ignores it (see PORTING_NOTES §5b).
   - **End of the 1st half:** the state becomes the Q3 kickoff to the 2nd-half receiver.
   - **End of game:** the outcome is valued exactly (1 or 0). A tie at the end of
     regulation means OT, which gets support 0.
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
5. **Recommendation:** the option with the highest EV, unless a refusal rule fires.

## Refusal rules (`DecisionConfig`, all configurable)
| rule | default | why |
|---|---|---|
| winner price outside [min_price, max_price] | [0.03, 0.97] | the response model loses to "no move" there (RESPONSE_MODEL.md) |
| price older than max_quote_age_s | 30 s | stale |
| bid/ask wider than max_bid_ask | 0.04 | wide market |
| n_trades below min_trades | 0 (off) | thin; quote age and width already cover this |
| halted | — | — |
| Kalshi spread or total ladder missing or quality < min_ladder_quality | 0.2 | no team-strength features, and no fallback to the betting line |
| qtr > max_qtr | 4 | OT clock logic is outside nfl4th's components |
| option support (probability-weighted) < min_support | 0.01 | outside the model's data |
| any outcome with prob ≥ 5% and support < min_outcome_support | 0.002 | as above |

A refused play gets `recommendation = None`, `no_recommendation = True` and a
`no_recommendation_reason`. Nothing is valued for it: there is no WP-model fallback.
`wp_baseline` is not imported by this module, and `tests/test_quarantine.py` enforces
that.

On all 3,997 4th-down plays in 2025, the defaults give:
- **recommendations:** punt 1,172; FG 820; go 545;
- **refused:** 1,460 (36.5%).

Refusal reasons (a play can have several):

| reason | plays |
|---|---|
| outcome states outside support | 624 |
| price outside [0.03, 0.97] | 535 |
| pregame ladder unusable | 154 spread / 115 total |
| stale price | 77 |
| wide market | 73 |
| no price | 31 |
| halted | 31 |
| OT | 24 |

The "too thin" count (340) was measured with min_trades = 1, before it was switched off.
Phase 8 re-measures the rates with the final defaults.

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
