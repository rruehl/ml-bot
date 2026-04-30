#!/usr/bin/env python3
"""
BTC 1-Minute Dual-Stream WebSocket Collector
=============================================
Connects to two WebSocket feeds concurrently and writes one row per minute to
a single CSV containing everything needed to train the 15-minute prediction model:

  Coinbase Advanced Trade (Spot):
    - 1-min OHLCV candle           (from market_trades stream)
    - Buy volume, sell volume, CVD  (exact taker flow)
    - Order book snapshot at t=0s   (minute open: spread, imbalance, depth)
    - Order book snapshot at t=30s  (intra-minute: captures book flips)
    - Intra-minute spread max

  Bybit V5 Linear Public (Perpetual Futures):
    - Mark price at t=0s and t=30s  (for spot-futures basis)
    - Funding rate                   (sentiment / crowding signal)
    - Open interest in USDT          (capital commitment)
    - Minutes to next funding event  (pre-funding unwind pressure)
    - Futures L1 bid-ask spread bps  (market quality)
    - Futures mark price 1h ago      (momentum baseline)
    - Futures order book imbalances L1/L5/L10/slope at t=0s and t=30s
    - Futures taker buy/sell volume per bar + running CVD
    - Futures bar high/low
    - Liquidation flow: long and short volumes per bar

Architecture: queue-based dual-producer / single-consumer.
  - coinbase_producer: Coinbase WS → queue
  - bybit_producer:    Bybit WS → queue (mutates BybitState in-producer for freshest snapshots)
  - consumer:          drains queue, owns all bar state, writes CSV

Usage:
    python btc_ws_collector.py
    python btc_ws_collector.py --output data/btc_1min.csv
    python btc_ws_collector.py --minutes 60    # stop after 60 bars (testing)
    python btc_ws_collector.py --product ETH-USD
"""

import argparse
import asyncio
import csv
import json
import logging
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import websockets
except ImportError:
    print("websockets is required:  pip install websockets")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WS_URL              = "wss://advanced-trade-ws.coinbase.com"
BYBIT_WS_URL        = "wss://stream.bybit.com/v5/public/linear"
BYBIT_PING_INTERVAL = 18          # Bybit requires app-level {"op":"ping"} every <20s
BOOK_DEPTH          = 50          # raw price levels per snapshot
QUEUE_MAXSIZE       = 2000        # ~10s buffer at Bybit 200ms tick rate
DEFAULT_OUT         = Path(__file__).resolve().parent.parent / "data" / "btc_1min.csv"
MAX_BACKOFF         = 60
WS_MAX_SIZE         = 16 * 1024 * 1024  # 16 MB — level2 snapshots exceed 1 MB default


def _level_cols(suffix: str) -> list[str]:
    """Raw Coinbase order book column names for one snapshot."""
    cols = []
    for i in range(BOOK_DEPTH):
        cols.append(f"bid[{i}].price_{suffix}")
        cols.append(f"bid[{i}].size_{suffix}")
    for i in range(BOOK_DEPTH):
        cols.append(f"ask[{i}].price_{suffix}")
        cols.append(f"ask[{i}].size_{suffix}")
    return cols


# All columns written to CSV — order matters.
COLUMNS = [
    "timestamp",
    # 1-min OHLCV
    "open", "high", "low", "close", "volume",
    # Taker flow (spot)
    "buy_volume", "sell_volume", "cvd", "trade_count",
    # ── Spot open snapshot (t = 0s) ─────────────────────────────────────────
    "best_bid_open", "best_ask_open", "spread_bps_open",
    "imbalance_l1_open", "imbalance_l5_open", "imbalance_l10_open", "imbalance_slope_open",
    "bid_depth_10_open", "ask_depth_10_open",
    *_level_cols("open"),
    # ── Spot close snapshot (t = +30s) ──────────────────────────────────────
    "best_bid_close", "best_ask_close", "spread_bps_close",
    "imbalance_l1_close", "imbalance_l5_close", "imbalance_l10_close", "imbalance_slope_close",
    "bid_depth_10_close", "ask_depth_10_close",
    *_level_cols("close"),
    # ── Spot intra-minute extreme ────────────────────────────────────────────
    "spread_bps_max",
    # ── Bybit perpetual futures ──────────────────────────────────────────────
    # Tickers (mark price, funding, OI — snapshotted at t=0s and t=30s)
    "bybit_mark_open",          # mark price at t=0s
    "bybit_mark_close",         # mark price at t=30s
    "bybit_funding_rate",       # current 8h funding rate
    "bybit_oi",                 # open interest in USDT
    "bybit_next_funding_min",   # minutes until next 8h funding event
    "bybit_futures_spread_bps", # L1 bid-ask spread bps (from bid1/ask1 in tickers)
    "bybit_prev_price_1h",      # futures mark price 1 hour ago
    # Futures order book imbalances
    "bybit_imbal_l1_open",   "bybit_imbal_l5_open",
    "bybit_imbal_l10_open",  "bybit_imbal_slope_open",
    "bybit_imbal_l1_close",  "bybit_imbal_l5_close",
    "bybit_imbal_l10_close", "bybit_imbal_slope_close",
    # Futures taker flow (per bar)
    "bybit_buy_volume",   # taker buy vol (BTC)
    "bybit_sell_volume",  # taker sell vol (BTC)
    "bybit_cvd",          # running lifetime CVD (BTC)
    "bybit_bar_high",     # futures bar high price
    "bybit_bar_low",      # futures bar low price
    # Liquidations (per bar)
    "bybit_liq_long_vol",   # long positions liquidated (BTC) — bearish cascade signal
    "bybit_liq_short_vol",  # short positions liquidated (BTC) — bullish cascade signal
]


# ---------------------------------------------------------------------------
# Typed event dataclasses
# ---------------------------------------------------------------------------
@dataclass
class CoinbaseEvent:
    channel: str
    msg: dict
    ts: datetime


@dataclass
class BybitEvent:
    topic: str      # "orderbook.50.BTCUSDT", "tickers.BTCUSDT", etc.
    msg_type: str   # "snapshot" or "delta"
    data: Any
    ts: datetime


# ---------------------------------------------------------------------------
# Coinbase spot order book
# ---------------------------------------------------------------------------
class OrderBook:
    """
    Maintains the current best order book state from Coinbase level2 WebSocket updates.

    Advanced Trade channel:
      - Snapshot:  {type: "snapshot", updates: [{side, price_level, new_quantity}, ...]}
      - Update:    {type: "update",   updates: [...]}  — new_quantity "0" = remove level
    """

    def __init__(self):
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.ready = False

    def apply_snapshot(self, updates: list) -> None:
        self.bids.clear()
        self.asks.clear()
        for u in updates:
            price = float(u["price_level"])
            size  = float(u["new_quantity"])
            if size > 0:
                if u["side"] == "bid":
                    self.bids[price] = size
                else:
                    self.asks[price] = size
        self.ready = True
        log.info("Coinbase book snapshot: %d bids, %d asks", len(self.bids), len(self.asks))

    def apply_update(self, updates: list) -> None:
        for u in updates:
            price = float(u["price_level"])
            size  = float(u["new_quantity"])
            book  = self.bids if u["side"] == "bid" else self.asks
            if size == 0.0:
                book.pop(price, None)
            else:
                book[price] = size

    def snapshot(self) -> dict:
        """Compute aggregates + raw levels. Returns NaN-filled dict if not ready."""
        nan = float("nan")
        _empty = {
            "best_bid": nan, "best_ask": nan, "spread_bps": nan,
            "imbalance_l1": nan, "imbalance_l5": nan, "imbalance_l10": nan,
            "imbalance_slope": nan,
            "bid_depth_10": nan, "ask_depth_10": nan,
            "raw_bids": [], "raw_asks": [],
        }
        if not self.ready or not self.bids or not self.asks:
            return _empty

        best_bid = max(self.bids)
        best_ask = min(self.asks)
        if best_bid >= best_ask:
            return _empty

        mid    = (best_bid + best_ask) / 2.0
        spread = (best_ask - best_bid) / mid * 10_000

        top_bids = sorted(self.bids.items(), reverse=True)[:BOOK_DEPTH]
        top_asks = sorted(self.asks.items())[:BOOK_DEPTH]

        def _imbal(bids_n, asks_n):
            b = sum(s for _, s in bids_n)
            a = sum(s for _, s in asks_n)
            t = b + a
            return (b - a) / t if t > 0 else 0.0

        imbal_l1  = _imbal(top_bids[:1],  top_asks[:1])
        imbal_l5  = _imbal(top_bids[:5],  top_asks[:5])
        imbal_l10 = _imbal(top_bids[:10], top_asks[:10])

        return {
            "best_bid":        best_bid,
            "best_ask":        best_ask,
            "spread_bps":      spread,
            "imbalance_l1":    imbal_l1,
            "imbalance_l5":    imbal_l5,
            "imbalance_l10":   imbal_l10,
            "imbalance_slope": imbal_l1 - imbal_l10,
            "bid_depth_10":    sum(s for _, s in top_bids[:10]),
            "ask_depth_10":    sum(s for _, s in top_asks[:10]),
            "raw_bids":        top_bids,
            "raw_asks":        top_asks,
        }


# ---------------------------------------------------------------------------
# Bybit futures order book
# ---------------------------------------------------------------------------
class BybitOrderBook:
    """
    Maintains a 50-level Bybit perpetual futures order book (V5 linear format).

    V5 snapshot/delta: {"b": [[price, size], ...], "a": [[price, size], ...]}
    size "0" means remove that level.

    Gap detection uses the per-symbol "u" (update ID) field, NOT "seq".
    Bybit's "seq" is a cross-market counter that increments for every event
    across all products — consecutive BTCUSDT updates will always have gaps
    in seq. The "u" field increments by 1 for each BTCUSDT orderbook update.
    """

    def __init__(self):
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.ready = False
        self._u: int | None = None   # per-symbol update ID, sequential

    def apply_snapshot(self, data: dict) -> None:
        self.bids.clear()
        self.asks.clear()
        for p, s in data.get("b", []):
            if (sz := float(s)) > 0:
                self.bids[float(p)] = sz
        for p, s in data.get("a", []):
            if (sz := float(s)) > 0:
                self.asks[float(p)] = sz
        self._u = data.get("u")
        self.ready = True
        log.info("Bybit book snapshot: %d bids, %d asks", len(self.bids), len(self.asks))

    def apply_delta(self, data: dict) -> None:
        new_u = data.get("u")
        if self._u is not None and new_u is not None:
            if new_u != self._u + 1:
                log.warning(
                    "Bybit update-ID gap: expected %d got %d — marking book stale",
                    self._u + 1, new_u,
                )
                self.ready = False
                self._u = new_u
                return
        self._u = new_u
        for p, s in data.get("b", []):
            price, size = float(p), float(s)
            if size == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = size
        for p, s in data.get("a", []):
            price, size = float(p), float(s)
            if size == 0.0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = size

    def snapshot_imbalances(self) -> dict:
        """Return L1/L5/L10 imbalances and slope. Returns NaN dict if not ready."""
        nan = float("nan")
        _empty = {
            "imbalance_l1": nan, "imbalance_l5": nan,
            "imbalance_l10": nan, "imbalance_slope": nan,
        }
        if not self.ready or not self.bids or not self.asks:
            return _empty
        if max(self.bids) >= min(self.asks):
            return _empty

        top_bids = sorted(self.bids.items(), reverse=True)[:50]
        top_asks = sorted(self.asks.items())[:50]

        def _imbal(bids_n, asks_n):
            bv = sum(s for _, s in bids_n)
            av = sum(s for _, s in asks_n)
            t  = bv + av
            return (bv - av) / t if t > 0 else 0.0

        il1  = _imbal(top_bids[:1],  top_asks[:1])
        il5  = _imbal(top_bids[:5],  top_asks[:5])
        il10 = _imbal(top_bids[:10], top_asks[:10])
        return {
            "imbalance_l1":    il1,
            "imbalance_l5":    il5,
            "imbalance_l10":   il10,
            "imbalance_slope": il1 - il10,
        }


# ---------------------------------------------------------------------------
# Bybit state — mutated in producer, snapshotted by consumer at bar boundaries
# ---------------------------------------------------------------------------
class BybitState:
    """
    Shared mutable state for all Bybit data streams.

    Updated by bybit_producer before each queue.put() so that consumer
    snapshots always reflect the absolute latest futures state at bar boundaries.

    Per-bar accumulators (buy_volume, sell_volume, bar_high, bar_low,
    liq_long_vol, liq_short_vol) are reset by consumer.reset_bar() at each
    bar flush. running_cvd is never reset.
    """

    def __init__(self):
        self.book = BybitOrderBook()

        # ── Ticker fields (updated on each tickers.BTCUSDT message) ──────────
        self.mark_price:    float | None = None
        self.funding_rate:  float | None = None
        self.open_interest: float | None = None
        self.next_funding_ms: int | None = None   # raw ms timestamp
        self.bid1_price:    float | None = None
        self.ask1_price:    float | None = None
        self.prev_price_1h: float | None = None

        # ── Per-bar accumulators (reset at each bar flush) ────────────────────
        self.bar_buy_volume:   float = 0.0
        self.bar_sell_volume:  float = 0.0
        self.bar_high:         float = float("-inf")
        self.bar_low:          float = float("inf")
        self.liq_long_vol:     float = 0.0   # "Buy" side liq = long position liquidated
        self.liq_short_vol:    float = 0.0   # "Sell" side liq = short position liquidated

        # ── Lifetime (never reset) ────────────────────────────────────────────
        self.running_cvd: float = 0.0

    def update_ticker(self, data: dict) -> None:
        """
        Parse tickers.BTCUSDT message. Bybit sends only changed fields in delta
        messages (empty string for unchanged fields), so we skip empty values.
        """
        def _parse(key: str) -> float | None:
            v = data.get(key, "")
            return float(v) if v else None

        def _parse_int(key: str) -> int | None:
            v = data.get(key, "")
            return int(v) if v else None

        mark = _parse("markPrice")
        if mark is not None:
            self.mark_price = mark
        fr = _parse("fundingRate")
        if fr is not None:
            self.funding_rate = fr
        oi = _parse("openInterestValue")
        if oi is not None:
            self.open_interest = oi
        nft = _parse_int("nextFundingTime")
        if nft is not None:
            self.next_funding_ms = nft
        b1 = _parse("bid1Price")
        if b1 is not None:
            self.bid1_price = b1
        a1 = _parse("ask1Price")
        if a1 is not None:
            self.ask1_price = a1
        p1h = _parse("prevPrice1h")
        if p1h is not None:
            self.prev_price_1h = p1h

    def add_trade(self, price: float, size: float, taker_side: str) -> None:
        """
        taker_side: "Buy"  → taker lifted the ask → CVD positive
                    "Sell" → taker hit the bid   → CVD negative
        """
        if taker_side == "Buy":
            self.bar_buy_volume += size
            self.running_cvd    += size
        else:
            self.bar_sell_volume += size
            self.running_cvd     -= size
        if price > self.bar_high:
            self.bar_high = price
        if price < self.bar_low:
            self.bar_low = price

    def add_liquidation(self, size: float, side: str) -> None:
        """
        side: "Buy"  → long position was liquidated (bearish cascade signal)
              "Sell" → short position was liquidated (bullish cascade signal)
        """
        if side == "Buy":
            self.liq_long_vol += size
        else:
            self.liq_short_vol += size

    def reset_bar(self) -> None:
        """Reset per-bar accumulators after flush. Never resets running_cvd."""
        self.bar_buy_volume  = 0.0
        self.bar_sell_volume = 0.0
        self.bar_high        = float("-inf")
        self.bar_low         = float("inf")
        self.liq_long_vol    = 0.0
        self.liq_short_vol   = 0.0

    def snapshot(self) -> dict:
        """
        Return a point-in-time snapshot of all ticker + book imbalance fields.
        Always returns float or NaN — never None.
        """
        nan = float("nan")
        spread_bps = nan
        if self.bid1_price and self.ask1_price and self.mark_price:
            spread_bps = (self.ask1_price - self.bid1_price) / self.mark_price * 10_000
        imbal = self.book.snapshot_imbalances()
        return {
            "mark_price":       self.mark_price    if self.mark_price    is not None else nan,
            "funding_rate":     self.funding_rate  if self.funding_rate  is not None else nan,
            "open_interest":    self.open_interest if self.open_interest is not None else nan,
            "futures_spread_bps": spread_bps,
            "prev_price_1h":    self.prev_price_1h if self.prev_price_1h is not None else nan,
            "imbalance_l1":     imbal["imbalance_l1"],
            "imbalance_l5":     imbal["imbalance_l5"],
            "imbalance_l10":    imbal["imbalance_l10"],
            "imbalance_slope":  imbal["imbalance_slope"],
        }

    def next_funding_minutes(self, now_ms: int) -> float:
        """Compute minutes until next 8h funding. Returns NaN if unknown."""
        if self.next_funding_ms is None:
            return float("nan")
        return round((self.next_funding_ms - now_ms) / 60_000.0, 2)


# ---------------------------------------------------------------------------
# Minute bar accumulator
# ---------------------------------------------------------------------------
class MinuteBar:
    """
    Accumulates one minute of data from both Coinbase (spot) and Bybit (futures).

    Spot data:
      - OHLCV + taker flow from the trade stream
      - Two spot order book snapshots (open at t=0s, close at t=30s)
      - Running intra-minute spread max

    Futures data:
      - Two BybitState snapshots (open at t=0s, close at t=30s)
      - Per-bar taker volumes, CVD, high/low, liquidations (from BybitState at flush)
    """

    def __init__(self, minute: datetime, open_snap: dict):
        self.minute    = minute
        self.open_snap = open_snap

        self.mid_snap_taken = False
        self.mid_snap: dict | None = None

        # Spot trade accumulators
        self.open_price:  float | None = None
        self.high_price:  float        = float("-inf")
        self.low_price:   float        = float("inf")
        self.close_price: float | None = None
        self.volume      = 0.0
        self.buy_volume  = 0.0
        self.sell_volume = 0.0
        self.trade_count = 0
        self._spread_max = float("-inf")

        # Bybit snapshots (set by consumer at t=0s and t=30s)
        self.bybit_open:  dict | None = None
        self.bybit_close: dict | None = None

        # Bybit per-bar values (captured from BybitState at bar flush)
        self.bybit_buy_volume:    float = 0.0
        self.bybit_sell_volume:   float = 0.0
        self.bybit_cvd:           float = 0.0
        self.bybit_bar_high:      float = float("nan")
        self.bybit_bar_low:       float = float("nan")
        self.bybit_liq_long_vol:  float = 0.0
        self.bybit_liq_short_vol: float = 0.0
        self.bybit_next_funding_min: float = float("nan")

    # ── incoming spot data ──────────────────────────────────────────────────

    def add_trade(self, price: float, size: float, taker_side: str) -> None:
        if self.open_price is None:
            self.open_price = price
        self.close_price  = price
        self.high_price   = max(self.high_price, price)
        self.low_price    = min(self.low_price,  price)
        self.volume      += size
        self.trade_count += 1
        if taker_side == "buy":
            self.buy_volume += size
        else:
            self.sell_volume += size

    def update_spread(self, spread_bps: float) -> None:
        if spread_bps == spread_bps:  # not NaN
            self._spread_max = max(self._spread_max, spread_bps)

    def take_mid_snapshot(self, snap: dict) -> None:
        self.mid_snap       = snap
        self.mid_snap_taken = True

    # ── incoming Bybit data ─────────────────────────────────────────────────

    def take_bybit_open_snapshot(self, snap: dict) -> None:
        self.bybit_open = snap

    def take_bybit_close_snapshot(self, snap: dict) -> None:
        self.bybit_close = snap

    # ── output ─────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten_levels(snap: dict, suffix: str) -> dict:
        """Flatten raw_bids / raw_asks into bid[i].price_suffix columns."""
        nan = float("nan")
        row: dict = {}
        bids = snap.get("raw_bids", [])
        asks = snap.get("raw_asks", [])
        for i in range(BOOK_DEPTH):
            row[f"bid[{i}].price_{suffix}"] = bids[i][0] if i < len(bids) else nan
            row[f"bid[{i}].size_{suffix}"]  = bids[i][1] if i < len(bids) else nan
        for i in range(BOOK_DEPTH):
            row[f"ask[{i}].price_{suffix}"] = asks[i][0] if i < len(asks) else nan
            row[f"ask[{i}].size_{suffix}"]  = asks[i][1] if i < len(asks) else nan
        return row

    def to_row(self, running_cvd: float) -> dict:
        nan = float("nan")
        o   = self.open_snap
        c   = self.mid_snap if self.mid_snap is not None else self.open_snap
        bo  = self.bybit_open  or {}
        bc  = self.bybit_close or {}

        row = {
            "timestamp":  self.minute.strftime("%Y-%m-%dT%H:%M:00Z"),
            "open":       self.open_price  if self.open_price  is not None else nan,
            "high":       self.high_price  if self.high_price  > float("-inf") else nan,
            "low":        self.low_price   if self.low_price   < float("inf")  else nan,
            "close":      self.close_price if self.close_price is not None else nan,
            "volume":     self.volume,
            "buy_volume": self.buy_volume,
            "sell_volume":self.sell_volume,
            "cvd":        running_cvd,
            "trade_count":self.trade_count,
            # ── Spot open snapshot (t=0s) ────────────────────────────────────
            "best_bid_open":         o.get("best_bid",        nan),
            "best_ask_open":         o.get("best_ask",        nan),
            "spread_bps_open":       o.get("spread_bps",      nan),
            "imbalance_l1_open":     o.get("imbalance_l1",    nan),
            "imbalance_l5_open":     o.get("imbalance_l5",    nan),
            "imbalance_l10_open":    o.get("imbalance_l10",   nan),
            "imbalance_slope_open":  o.get("imbalance_slope", nan),
            "bid_depth_10_open":     o.get("bid_depth_10",    nan),
            "ask_depth_10_open":     o.get("ask_depth_10",    nan),
            # ── Spot close snapshot (t=+30s) ─────────────────────────────────
            "best_bid_close":        c.get("best_bid",        nan),
            "best_ask_close":        c.get("best_ask",        nan),
            "spread_bps_close":      c.get("spread_bps",      nan),
            "imbalance_l1_close":    c.get("imbalance_l1",    nan),
            "imbalance_l5_close":    c.get("imbalance_l5",    nan),
            "imbalance_l10_close":   c.get("imbalance_l10",   nan),
            "imbalance_slope_close": c.get("imbalance_slope", nan),
            "bid_depth_10_close":    c.get("bid_depth_10",    nan),
            "ask_depth_10_close":    c.get("ask_depth_10",    nan),
            # ── Spot intra-minute extreme ────────────────────────────────────
            "spread_bps_max": self._spread_max if self._spread_max > float("-inf") else nan,
            # ── Bybit tickers (close snapshot = most recent) ─────────────────
            "bybit_mark_open":          bo.get("mark_price",        nan),
            "bybit_mark_close":         bc.get("mark_price",        nan),
            "bybit_funding_rate":       bc.get("funding_rate",      nan),
            "bybit_oi":                 bc.get("open_interest",     nan),
            "bybit_next_funding_min":   self.bybit_next_funding_min,
            "bybit_futures_spread_bps": bc.get("futures_spread_bps",nan),
            "bybit_prev_price_1h":      bc.get("prev_price_1h",     nan),
            # ── Bybit order book imbalances ──────────────────────────────────
            "bybit_imbal_l1_open":     bo.get("imbalance_l1",    nan),
            "bybit_imbal_l5_open":     bo.get("imbalance_l5",    nan),
            "bybit_imbal_l10_open":    bo.get("imbalance_l10",   nan),
            "bybit_imbal_slope_open":  bo.get("imbalance_slope", nan),
            "bybit_imbal_l1_close":    bc.get("imbalance_l1",    nan),
            "bybit_imbal_l5_close":    bc.get("imbalance_l5",    nan),
            "bybit_imbal_l10_close":   bc.get("imbalance_l10",   nan),
            "bybit_imbal_slope_close": bc.get("imbalance_slope", nan),
            # ── Bybit per-bar taker flow ─────────────────────────────────────
            "bybit_buy_volume":  self.bybit_buy_volume,
            "bybit_sell_volume": self.bybit_sell_volume,
            "bybit_cvd":         self.bybit_cvd,
            "bybit_bar_high":    self.bybit_bar_high,
            "bybit_bar_low":     self.bybit_bar_low,
            # ── Bybit per-bar liquidations ───────────────────────────────────
            "bybit_liq_long_vol":  self.bybit_liq_long_vol,
            "bybit_liq_short_vol": self.bybit_liq_short_vol,
        }
        # Spot raw per-level data
        row.update(self._flatten_levels(o, "open"))
        row.update(self._flatten_levels(c, "close"))
        return row


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------
class CsvWriter:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists() or path.stat().st_size == 0
        self._fh     = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=COLUMNS)
        if is_new:
            self._writer.writeheader()
            log.info("Created %s  (%d columns)", path, len(COLUMNS))
        else:
            log.info("Appending to %s  (%d columns)", path, len(COLUMNS))

    def write(self, row: dict) -> None:
        self._writer.writerow({k: row.get(k, "") for k in COLUMNS})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Coinbase producer
# ---------------------------------------------------------------------------
async def coinbase_producer(
    queue: asyncio.Queue,
    product_id: str,
    shutdown: asyncio.Event,
) -> None:
    """
    Connects to Coinbase Advanced Trade WebSocket and puts CoinbaseEvent objects
    into the shared queue. Independent reconnect loop — Coinbase failures do NOT
    affect the Bybit producer.
    """
    sub_l2 = json.dumps({
        "type": "subscribe", "product_ids": [product_id], "channel": "level2",
    })
    sub_trades = json.dumps({
        "type": "subscribe", "product_ids": [product_id], "channel": "market_trades",
    })
    backoff = 1

    while not shutdown.is_set():
        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=15,
                close_timeout=5,
                max_size=WS_MAX_SIZE,
            ) as ws:
                await ws.send(sub_l2)
                await ws.send(sub_trades)
                log.info("Coinbase connected (%s)", product_id)
                backoff = 1

                async for raw_msg in ws:
                    if shutdown.is_set():
                        return
                    msg = json.loads(raw_msg)
                    now = datetime.now(timezone.utc)
                    await queue.put(CoinbaseEvent(
                        channel=msg.get("channel", ""),
                        msg=msg,
                        ts=now,
                    ))

        except (websockets.ConnectionClosed, ConnectionError, OSError) as exc:
            if shutdown.is_set():
                return
            log.warning("Coinbase lost (%s) — retry in %ds", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
        except asyncio.CancelledError:
            return


# ---------------------------------------------------------------------------
# Bybit ping loop
# ---------------------------------------------------------------------------
async def _bybit_ping_loop(ws, shutdown: asyncio.Event) -> None:
    """
    Sends Bybit application-level ping every BYBIT_PING_INTERVAL seconds.
    Bybit requires {"op":"ping"} — it does NOT support RFC 6455 protocol pings.
    This task is tied to the WebSocket connection lifetime and is cancelled when
    the connection closes.
    """
    ping_msg = json.dumps({"op": "ping"})
    while not shutdown.is_set():
        await asyncio.sleep(BYBIT_PING_INTERVAL)
        try:
            await ws.send(ping_msg)
            log.debug("Bybit ping sent")
        except websockets.ConnectionClosed:
            return


# ---------------------------------------------------------------------------
# Bybit producer
# ---------------------------------------------------------------------------
async def bybit_producer(
    queue: asyncio.Queue,
    bybit_state: BybitState,
    shutdown: asyncio.Event,
) -> None:
    """
    Connects to Bybit V5 public linear WebSocket and handles 4 topics:
      - orderbook.50.BTCUSDT  → BybitOrderBook (snapshot + delta)
      - tickers.BTCUSDT       → mark price, funding, OI, bid1/ask1, prevPrice1h
      - publicTrade.BTCUSDT   → per-trade buy/sell volumes, CVD, bar high/low
      - allLiquidation.BTCUSDT→ long/short liquidation volumes

    BybitState is mutated BEFORE queue.put() so consumer snapshots at bar
    boundaries always reflect the absolute latest futures data.

    ping_interval=None: Bybit rejects RFC 6455 pings — use app-level ping task.
    """
    sub_msg = json.dumps({
        "op": "subscribe",
        "args": [
            "orderbook.50.BTCUSDT",
            "tickers.BTCUSDT",
            "publicTrade.BTCUSDT",
            "allLiquidation.BTCUSDT",
        ],
    })
    backoff = 1

    while not shutdown.is_set():
        try:
            async with websockets.connect(
                BYBIT_WS_URL,
                ping_interval=None,   # must be None — Bybit rejects RFC 6455 pings
                ping_timeout=None,
                close_timeout=5,
                max_size=WS_MAX_SIZE,
            ) as ws:
                await ws.send(sub_msg)
                log.info("Bybit connected (orderbook.50 + tickers + publicTrade + allLiquidation)")
                backoff = 1

                ping_task = asyncio.create_task(
                    _bybit_ping_loop(ws, shutdown),
                    name="bybit-ping",
                )
                try:
                    async for raw_msg in ws:
                        if shutdown.is_set():
                            return

                        msg = json.loads(raw_msg)

                        # Skip op confirmations (subscribe ack, pong)
                        if "op" in msg:
                            continue
                        if "topic" not in msg:
                            continue

                        topic    = msg["topic"]
                        msg_type = msg.get("type", "delta")
                        data     = msg.get("data", {})
                        now      = datetime.now(timezone.utc)

                        # ── Mutate BybitState BEFORE queuing ─────────────────
                        if topic == "tickers.BTCUSDT":
                            bybit_state.update_ticker(data)

                        elif topic == "orderbook.50.BTCUSDT":
                            if msg_type == "snapshot":
                                bybit_state.book.apply_snapshot(data)
                            else:
                                bybit_state.book.apply_delta(data)

                        elif topic == "publicTrade.BTCUSDT":
                            for t in (data if isinstance(data, list) else []):
                                try:
                                    bybit_state.add_trade(
                                        float(t["p"]), float(t["v"]), t["S"]
                                    )
                                except (KeyError, ValueError):
                                    pass

                        elif topic == "allLiquidation.BTCUSDT":
                            for t in (data if isinstance(data, list) else [data]):
                                try:
                                    bybit_state.add_liquidation(float(t["v"]), t["S"])
                                except (KeyError, ValueError):
                                    pass

                        await queue.put(BybitEvent(
                            topic=topic,
                            msg_type=msg_type,
                            data=data,
                            ts=now,
                        ))

                finally:
                    ping_task.cancel()
                    try:
                        await ping_task
                    except asyncio.CancelledError:
                        pass

        except (websockets.ConnectionClosed, ConnectionError, OSError) as exc:
            if shutdown.is_set():
                return
            log.warning("Bybit lost (%s) — retry in %ds", exc, backoff)
            bybit_state.book.ready = False
            bybit_state.reset_bar()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
        except asyncio.CancelledError:
            return


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------
async def consumer(
    queue: asyncio.Queue,
    bybit_state: BybitState,
    writer: CsvWriter,
    max_minutes: int | None,
    shutdown: asyncio.Event,
) -> None:
    """
    Single sequential consumer of all WebSocket events from both exchanges.
    Owns all bar state mutations and CSV writes. BybitState has already been
    updated by the producer before each event is queued.
    """
    book             = OrderBook()
    running_cvd      = 0.0
    current_bar: MinuteBar | None = None
    minutes_written  = 0

    while not shutdown.is_set():
        try:
            event = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        now    = event.ts
        minute = now.replace(second=0, microsecond=0)
        secs   = now.second + now.microsecond / 1_000_000

        if isinstance(event, CoinbaseEvent):
            channel = event.channel
            msg     = event.msg

            if channel == "l2_data":
                for ev in msg.get("events", []):
                    evt_type = ev.get("type")
                    updates  = ev.get("updates", [])

                    if evt_type == "snapshot":
                        book.apply_snapshot(updates)
                        if current_bar is None:
                            snap        = book.snapshot()
                            current_bar = MinuteBar(minute, snap)
                            current_bar.take_bybit_open_snapshot(bybit_state.snapshot())

                    elif evt_type == "update":
                        book.apply_update(updates)
                        snap = book.snapshot()

                        # ── New minute: flush completed bar ───────────────────
                        if current_bar is not None and minute > current_bar.minute:
                            now_ms = int(now.timestamp() * 1000)
                            current_bar.bybit_buy_volume    = bybit_state.bar_buy_volume
                            current_bar.bybit_sell_volume   = bybit_state.bar_sell_volume
                            current_bar.bybit_cvd           = bybit_state.running_cvd
                            current_bar.bybit_bar_high      = (
                                bybit_state.bar_high
                                if bybit_state.bar_high > float("-inf") else float("nan")
                            )
                            current_bar.bybit_bar_low       = (
                                bybit_state.bar_low
                                if bybit_state.bar_low  < float("inf")  else float("nan")
                            )
                            current_bar.bybit_liq_long_vol  = bybit_state.liq_long_vol
                            current_bar.bybit_liq_short_vol = bybit_state.liq_short_vol
                            current_bar.bybit_next_funding_min = bybit_state.next_funding_minutes(now_ms)
                            bybit_state.reset_bar()

                            row = current_bar.to_row(running_cvd)
                            writer.write(row)
                            minutes_written += 1
                            log.info(
                                "Bar %s  close=%-10s  vol=%.4f  cvd=%+.4f"
                                "  bybit_basis=%.5f  funding=%.6f",
                                row["timestamp"],
                                f"{row['close']:.2f}" if row["close"] == row["close"] else "NaN",
                                row["volume"],
                                row["cvd"],
                                (row["bybit_mark_close"] - row["close"]) / row["close"]
                                if row["bybit_mark_close"] == row["bybit_mark_close"]
                                   and row["close"] == row["close"]
                                   and row["close"] != 0
                                else float("nan"),
                                row["bybit_funding_rate"]
                                if row["bybit_funding_rate"] == row["bybit_funding_rate"]
                                else float("nan"),
                            )
                            if max_minutes is not None and minutes_written >= max_minutes:
                                log.info("Reached --minutes %d limit.", max_minutes)
                                shutdown.set()
                                break

                            current_bar = MinuteBar(minute, snap)
                            current_bar.take_bybit_open_snapshot(bybit_state.snapshot())

                        if current_bar is not None:
                            if not current_bar.mid_snap_taken and secs >= 30:
                                current_bar.take_mid_snapshot(snap)
                                current_bar.take_bybit_close_snapshot(bybit_state.snapshot())
                                log.debug("Mid-minute snapshot taken at +%.1fs", secs)
                            current_bar.update_spread(snap["spread_bps"])

            elif channel == "market_trades":
                for ev in msg.get("events", []):
                    for trade in ev.get("trades", []):
                        taker_side = trade.get("side", "").upper()
                        price      = float(trade["price"])
                        size       = float(trade["size"])
                        if taker_side == "BUY":
                            running_cvd += size
                            side_key     = "buy"
                        else:
                            running_cvd -= size
                            side_key     = "sell"
                        if current_bar is not None:
                            current_bar.add_trade(price, size, side_key)

            elif channel == "subscriptions":
                log.info("Coinbase subscriptions: %s", msg.get("events", ""))
            elif msg.get("type") == "error":
                log.error("Coinbase error: %s", msg.get("message"))

        # BybitEvent: state was already mutated in the producer — no-op here.
        # This branch exists for future per-event hooks (latency monitoring, etc.)

        queue.task_done()


# ---------------------------------------------------------------------------
# Collect — main entry point
# ---------------------------------------------------------------------------
async def collect(product_id: str, output: Path, max_minutes: int | None) -> None:
    """
    Dual-stream collection: Coinbase spot + Bybit perpetual futures.
    Queue-based architecture prevents either producer from blocking the other.
    return_exceptions=True: an unrecoverable Bybit crash does not cancel Coinbase.
    """
    queue       = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    bybit_state = BybitState()
    writer      = CsvWriter(output)
    shutdown    = asyncio.Event()

    def _signal_handler(sig, frame):
        log.info("Shutdown signal received — stopping after current bar.")
        shutdown.set()

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    log.info("Starting dual-stream collector: Coinbase %s + Bybit BTCUSDT perp", product_id)
    log.info("Output: %s  (%d columns)", output, len(COLUMNS))

    results = await asyncio.gather(
        coinbase_producer(queue, product_id, shutdown),
        bybit_producer(queue, bybit_state, shutdown),
        consumer(queue, bybit_state, writer, max_minutes, shutdown),
        return_exceptions=True,
    )

    writer.close()

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            names = ["coinbase_producer", "bybit_producer", "consumer"]
            log.error("Task %s raised: %s", names[i], result)

    log.info("Collector stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="BTC 1-minute dual-stream WebSocket collector (Coinbase + Bybit)"
    )
    parser.add_argument("--product",  default="BTC-USD",       help="Coinbase product ID")
    parser.add_argument("--output",   default=str(DEFAULT_OUT), help="Output CSV path")
    parser.add_argument("--minutes",  type=int, default=None,   help="Stop after N minute bars")
    args = parser.parse_args()

    asyncio.run(collect(
        product_id=args.product,
        output=Path(args.output),
        max_minutes=args.minutes,
    ))


if __name__ == "__main__":
    main()
