"""Orderbook math for Kalshi binary markets and winner-market normalization.

Kalshi books list bids only (`yes_dollars`, `no_dollars`: [[price, size], ...]
ascending). A NO bid at q is a YES ask at 1 - q, so YES asks are derived from NO bids.
A book with no bids on one side has no ask (or no bid) and therefore no midpoint; it
is reported as such rather than imputed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Quote:
    bid: float | None  # best YES bid
    ask: float | None  # best YES ask = 1 - best NO bid
    bid_size: float
    ask_size: float
    bid_depth: float  # YES-bid contracts within `depth_window` of the best bid
    ask_depth: float

    @property
    def two_sided(self) -> bool:
        return self.bid is not None and self.ask is not None

    @property
    def mid(self) -> float | None:
        return (self.bid + self.ask) / 2 if self.two_sided else None

    @property
    def spread(self) -> float | None:
        return self.ask - self.bid if self.two_sided else None

    @property
    def microprice(self) -> float | None:
        """Size-weighted mid: leans toward the side with less resting size."""
        if not self.two_sided or self.bid_size + self.ask_size <= 0:
            return self.mid
        w = self.bid_size / (self.bid_size + self.ask_size)
        return w * self.ask + (1 - w) * self.bid

    def flipped(self) -> Quote:
        """The same book seen as the NO side (i.e. the other outcome)."""
        return Quote(
            bid=None if self.ask is None else 1 - self.ask,
            ask=None if self.bid is None else 1 - self.bid,
            bid_size=self.ask_size,
            ask_size=self.bid_size,
            bid_depth=self.ask_depth,
            ask_depth=self.bid_depth,
        )


def _levels(raw: list | None) -> list[tuple[float, float]]:
    out = [(float(p), float(s)) for p, s in (raw or []) if float(s) > 0]
    return sorted(out)


def quote_from_orderbook(book: dict, depth_window: float = 0.03) -> Quote:
    """Build a Quote from a REST orderbook response (`orderbook_fp` or `orderbook`)."""
    ob = book.get("orderbook_fp") or book.get("orderbook") or book
    yes = _levels(ob.get("yes_dollars") or ob.get("yes_dollars_fp") or ob.get("yes"))
    no = _levels(ob.get("no_dollars") or ob.get("no_dollars_fp") or ob.get("no"))
    bid, bid_size = (yes[-1] if yes else (None, 0.0))
    nbid, ask_size = (no[-1] if no else (None, 0.0))
    ask = None if nbid is None else round(1 - nbid, 6)
    bid_depth = sum(s for p, s in yes if bid is not None and p >= bid - depth_window - 1e-9)
    ask_depth = sum(s for p, s in no if nbid is not None and p >= nbid - depth_window - 1e-9)
    return Quote(bid, ask, bid_size, ask_size, bid_depth, ask_depth)


def posteam_win_price(team_quote: Quote | None, opp_quote: Quote | None) -> tuple[float | None, str]:
    """P(posteam wins) (+ half the tie probability: ties settle at $0.50) from the
    per-team winner markets.

    Uses the average of the posteam market's mid and 1 - the opponent market's mid
    when both are two-sided, otherwise whichever exists. Returns (price, source).
    """
    a = team_quote.mid if team_quote else None
    b = (1 - opp_quote.mid) if opp_quote and opp_quote.mid is not None else None
    if a is not None and b is not None:
        return (a + b) / 2, "both"
    if a is not None:
        return a, "team"
    if b is not None:
        return b, "opponent"
    return None, "none"


def logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))
