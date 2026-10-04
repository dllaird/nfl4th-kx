import pytest

from nfl4th_py.market.book import posteam_win_price, quote_from_orderbook

BOOK = {
    "orderbook_fp": {
        "no_dollars": [["0.3800", "263.89"], ["0.3900", "10001.21"], ["0.4000", "441.58"]],
        "yes_dollars": [["0.5700", "5.00"], ["0.5800", "10035.53"], ["0.5900", "7.18"]],
    }
}


def test_ask_derived_from_no_bids():
    q = quote_from_orderbook(BOOK)
    assert q.bid == pytest.approx(0.59)
    assert q.ask == pytest.approx(0.60)
    assert q.mid == pytest.approx(0.595)
    assert q.spread == pytest.approx(0.01)
    assert q.bid_size == pytest.approx(7.18) and q.ask_size == pytest.approx(441.58)
    assert q.bid_depth == pytest.approx(5 + 10035.53 + 7.18)


def test_bid_only_book_has_no_mid():
    q = quote_from_orderbook({"orderbook_fp": {"yes_dollars": [["0.20", "10"]], "no_dollars": []}})
    assert q.bid == pytest.approx(0.2) and q.ask is None
    assert q.mid is None and q.spread is None and not q.two_sided


def test_empty_book():
    q = quote_from_orderbook({"orderbook_fp": {"yes_dollars": None, "no_dollars": None}})
    assert q.mid is None


def test_flipped_is_other_outcome():
    q = quote_from_orderbook(BOOK).flipped()
    assert q.bid == pytest.approx(0.40) and q.ask == pytest.approx(0.41)


def test_posteam_price_combines_both_markets():
    team = quote_from_orderbook(BOOK)  # mid 0.595
    opp = quote_from_orderbook(
        {"orderbook_fp": {"yes_dollars": [["0.40", "5"]], "no_dollars": [["0.58", "5"]]}}
    )  # mid 0.41 -> 0.59 for posteam
    price, src = posteam_win_price(team, opp)
    assert src == "both" and price == pytest.approx((0.595 + 0.59) / 2)
    assert posteam_win_price(None, opp) == (pytest.approx(0.59), "opponent")
    assert posteam_win_price(None, None) == (None, "none")
