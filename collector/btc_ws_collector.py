#!/usr/bin/env python3
"""
BTC 1-Minute WebSocket Data Collector
=======================================
Connects to the Coinbase Exchange public WebSocket feed and writes one row
per minute to a single CSV file containing everything needed to train the
15-minute prediction model:

  - 1-min OHLCV candle           (from market_trades / matches stream)
  - Buy volume, sell volume, CVD  (exact taker flow — not a proxy)
  - Order book snapshot at t=0s   (minute open: spread, imbalance, depth)
  - Order book snapshot at t=30s  (intra-minute: captures book flips)
  - Intra-minute spread max       (captures liquidity spikes)

Replaces both coinbase_data_collector.py and update_macro_data.py.

Usage:
    pip install websockets
    python btc_ws_collector.py                        # write to data/btc_1min.csv
    python btc_ws_collector.py --output data/btc_1min.csv
    python btc_ws_collector.py --minutes 60           # stop after 60 bars (testing)
    python btc_ws_collector.py --product ETH-USD      # different product
"""

import argparse
import asyncio
import csv
import json
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

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
WS_URL      = "wss://advanced-trade-ws.coinbase.com"
BOOK_DEPTH  = 50   # raw price levels stored per snapshot (matches paper)
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "btc_1min.csv"
MAX_BACKOFF = 60   # seconds, cap for reconnection wait
WS_MAX_SIZE = 16 * 1024 * 1024  # 16 MB — level2 snapshots exceed the 1 MB default


def _level_cols(suffix: str) -> list[str]:
    """Raw order book column names for one snapshot side-by-side: bid[0].price_open …"""
    cols = []
    for i in range(BOOK_DEPTH):
        cols.append(f"bid[{i}].price_{suffix}")
        cols.append(f"bid[{i}].size_{suffix}")
    for i in range(BOOK_DEPTH):
        cols.append(f"ask[{i}].price_{suffix}")
        cols.append(f"ask[{i}].size_{suffix}")
    return cols


# All columns written to CSV — order matters for readability.
COLUMNS = [
    "timestamp",
    # 1-min OHLCV
    "open", "high", "low", "close", "volume",
    # Taker flow
    "buy_volume", "sell_volume", "cvd", "trade_count",
    # ── Open snapshot (t = 0s) ──────────────────────────────────────────
    # Aggregates
    "best_bid_open", "best_ask_open", "spread_bps_open",
    "imbalance_l1_open",    # level-1 only  (immediate pressure)
    "imbalance_l5_open",    # top-5 levels
    "imbalance_l10_open",   # top-10 levels
    "imbalance_slope_open", # l1 - l10 (top-heavy vs distributed)
    "bid_depth_10_open", "ask_depth_10_open",
    # Raw 50-level book — individual features per paper (81% of signal)
    *_level_cols("open"),
    # ── Close snapshot (t = +30s) ───────────────────────────────────────
    # Aggregates
    "best_bid_close", "best_ask_close", "spread_bps_close",
    "imbalance_l1_close",
    "imbalance_l5_close",
    "imbalance_l10_close",
    "imbalance_slope_close",
    "bid_depth_10_close", "ask_depth_10_close",
    # Raw 50-level book
    *_level_cols("close"),
    # ── Intra-minute extremes ────────────────────────────────────────────
    "spread_bps_max",
]


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------
class OrderBook:
    """
    Maintains the current best order book state from level2 WebSocket updates.

    Coinbase level2 channel:
      - First message: full snapshot  {type: "snapshot", bids: [[p,s],...], asks: [[p,s],...]}
      - Subsequent:    diffs          {type: "l2update",  changes: [[side,p,s],...]}
        where side="buy" means bid side, side="sell" means ask side.
        size="0" means remove that price level.
    """

    def __init__(self):
        self.bids: dict[float, float] = {}  # price → size
        self.asks: dict[float, float] = {}
        self.ready = False

    def apply_snapshot(self, updates: list) -> None:
        """
        Advanced Trade l2_data snapshot: list of
        {"side": "bid"/"offer", "price_level": "...", "new_quantity": "..."}
        """
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
        log.info("Book snapshot: %d bids, %d asks", len(self.bids), len(self.asks))

    def apply_update(self, updates: list) -> None:
        """
        Advanced Trade l2_data update: same structure as snapshot entries.
        new_quantity == "0" means remove the level.
        """
        for u in updates:
            price = float(u["price_level"])
            size  = float(u["new_quantity"])
            book  = self.bids if u["side"] == "bid" else self.asks
            if size == 0.0:
                book.pop(price, None)
            else:
                book[price] = size

    def snapshot(self) -> dict:
        """
        Compute features from the current book state.
        Returns NaN-filled dict if the book is not yet ready.

        Imbalance levels:
          imbalance_l1    — best bid size vs best ask size only (immediate pressure)
          imbalance_l5    — top 5 levels each side (medium depth)
          imbalance_l10   — top 10 levels each side (deep book)
          imbalance_slope — l1 minus l10: positive = pressure concentrated at top
                            (thin book); negative = more depth than the top shows
        """
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
            # Crossed book — data glitch, skip
            return _empty

        mid    = (best_bid + best_ask) / 2.0
        spread = (best_ask - best_bid) / mid * 10_000  # basis points

        # Sort once — used for both aggregates and raw level output
        top_bids = sorted(self.bids.items(), reverse=True)[:BOOK_DEPTH]
        top_asks = sorted(self.asks.items())[:BOOK_DEPTH]

        def _imbal(bids_n: list, asks_n: list) -> float:
            b = sum(s for _, s in bids_n)
            a = sum(s for _, s in asks_n)
            t = b + a
            return (b - a) / t if t > 0 else 0.0

        imbal_l1  = _imbal(top_bids[:1],  top_asks[:1])
        imbal_l5  = _imbal(top_bids[:5],  top_asks[:5])
        imbal_l10 = _imbal(top_bids,      top_asks)   # already capped at BOOK_DEPTH

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
            # Raw per-level data — individual features per paper (81% of signal)
            "raw_bids":        top_bids,   # list of (price, size), best first
            "raw_asks":        top_asks,   # list of (price, size), best first
        }


# ---------------------------------------------------------------------------
# Minute bar accumulator
# ---------------------------------------------------------------------------
class MinuteBar:
    """
    Accumulates one minute of data:
      - OHLCV + taker flow from the trade stream
      - Two order book snapshots (open at t=0s, close at t=30s)
      - Running intra-minute spread max
    """

    def __init__(self, minute: datetime, open_snap: dict):
        self.minute    = minute
        self.open_snap = open_snap

        self.mid_snap_taken = False
        self.mid_snap: dict | None = None

        # Trade accumulators
        self.open_price:  float | None = None
        self.high_price:  float        = float("-inf")
        self.low_price:   float        = float("inf")
        self.close_price: float | None = None
        self.volume      = 0.0
        self.buy_volume  = 0.0
        self.sell_volume = 0.0
        self.trade_count = 0

        # Spread tracking
        self._spread_max = float("-inf")

    # ── incoming data ──────────────────────────────────────────────────────

    def add_trade(self, price: float, size: float, taker_side: str) -> None:
        """
        taker_side: "buy"  → taker lifted the ask  → CVD positive contribution
                    "sell" → taker hit the bid      → CVD negative contribution
        """
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

        row = {
            "timestamp":             self.minute.strftime("%Y-%m-%dT%H:%M:00Z"),
            "open":                  self.open_price  if self.open_price  is not None else nan,
            "high":                  self.high_price  if self.high_price  > float("-inf") else nan,
            "low":                   self.low_price   if self.low_price   < float("inf")  else nan,
            "close":                 self.close_price if self.close_price is not None else nan,
            "volume":                self.volume,
            "buy_volume":            self.buy_volume,
            "sell_volume":           self.sell_volume,
            "cvd":                   running_cvd,
            "trade_count":           self.trade_count,
            # open snap aggregates (t = 0s)
            "best_bid_open":         o["best_bid"],
            "best_ask_open":         o["best_ask"],
            "spread_bps_open":       o["spread_bps"],
            "imbalance_l1_open":     o["imbalance_l1"],
            "imbalance_l5_open":     o["imbalance_l5"],
            "imbalance_l10_open":    o["imbalance_l10"],
            "imbalance_slope_open":  o["imbalance_slope"],
            "bid_depth_10_open":     o["bid_depth_10"],
            "ask_depth_10_open":     o["ask_depth_10"],
            # close snap aggregates (t = +30s)
            "best_bid_close":        c["best_bid"],
            "best_ask_close":        c["best_ask"],
            "spread_bps_close":      c["spread_bps"],
            "imbalance_l1_close":    c["imbalance_l1"],
            "imbalance_l5_close":    c["imbalance_l5"],
            "imbalance_l10_close":   c["imbalance_l10"],
            "imbalance_slope_close": c["imbalance_slope"],
            "bid_depth_10_close":    c["bid_depth_10"],
            "ask_depth_10_close":    c["ask_depth_10"],
            # intra-minute extreme
            "spread_bps_max":        self._spread_max if self._spread_max > float("-inf") else nan,
        }
        # Raw per-level data for both snapshots
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
            log.info("Created %s", path)
        else:
            log.info("Appending to %s", path)

    def write(self, row: dict) -> None:
        self._writer.writerow({k: row.get(k, "") for k in COLUMNS})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------
async def collect(product_id: str, output: Path, max_minutes: int | None) -> None:
    """
    Main collection loop with automatic reconnection on WebSocket errors.

    CVD note: Coinbase match messages report the MAKER side.
      maker side = "sell" → taker bought (lifted the ask) → CVD += size
      maker side = "buy"  → taker sold  (hit the bid)     → CVD -= size
    """
    book            = OrderBook()
    writer          = CsvWriter(output)
    running_cvd     = 0.0
    current_bar: MinuteBar | None = None
    minutes_written = 0
    shutdown        = asyncio.Event()

    def _signal_handler(sig, frame):
        log.info("Shutdown signal — stopping after current bar.")
        shutdown.set()

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Advanced Trade WebSocket requires one subscribe message per channel.
    sub_l2 = json.dumps({
        "type":        "subscribe",
        "product_ids": [product_id],
        "channel":     "level2",
    })
    sub_trades = json.dumps({
        "type":        "subscribe",
        "product_ids": [product_id],
        "channel":     "market_trades",
    })

    backoff = 1  # reconnection delay in seconds

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
                log.info("Connected and subscribed to %s (level2 + market_trades)", product_id)
                backoff = 1  # reset on successful connect

                async for raw_msg in ws:
                    if shutdown.is_set():
                        break

                    msg     = json.loads(raw_msg)
                    channel = msg.get("channel")
                    now     = datetime.now(timezone.utc)
                    minute  = now.replace(second=0, microsecond=0)
                    secs    = now.second + now.microsecond / 1_000_000

                    # ── Order book (l2_data channel) ───────────────────────
                    if channel == "l2_data":
                        for event in msg.get("events", []):
                            evt_type = event.get("type")
                            updates  = event.get("updates", [])

                            if evt_type == "snapshot":
                                book.apply_snapshot(updates)
                                if current_bar is None:
                                    current_bar = MinuteBar(minute, book.snapshot())

                            elif evt_type == "update":
                                book.apply_update(updates)
                                snap = book.snapshot()

                                # New minute rolled — flush completed bar
                                if current_bar is not None and minute > current_bar.minute:
                                    row = current_bar.to_row(running_cvd)
                                    writer.write(row)
                                    minutes_written += 1
                                    log.info(
                                        "Bar %s  close=%-10s  vol=%.4f  cvd=%+.4f  imbal=%+.3f",
                                        row["timestamp"],
                                        f"{row['close']:.2f}" if row["close"] == row["close"] else "NaN",
                                        row["volume"],
                                        row["cvd"],
                                        row["imbalance_l10_close"] if row["imbalance_l10_close"] == row["imbalance_l10_close"] else 0,
                                    )
                                    if max_minutes is not None and minutes_written >= max_minutes:
                                        log.info("Reached --minutes %d limit.", max_minutes)
                                        shutdown.set()
                                        break
                                    current_bar = MinuteBar(minute, snap)

                                if current_bar is not None:
                                    if not current_bar.mid_snap_taken and secs >= 30:
                                        current_bar.take_mid_snapshot(snap)
                                        log.debug("Mid-minute snapshot taken at +%.1fs", secs)
                                    current_bar.update_spread(snap["spread_bps"])

                    # ── Trades (market_trades channel) ─────────────────────
                    elif channel == "market_trades":
                        for event in msg.get("events", []):
                            for trade in event.get("trades", []):
                                # Advanced Trade reports the TAKER side directly
                                # (uppercase "BUY" / "SELL"), opposite of legacy "matches"
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
                        log.info("Subscriptions confirmed: %s", msg.get("events", ""))

                    elif msg.get("type") == "error":
                        log.error("WebSocket error from server: %s", msg.get("message"))

        except (websockets.ConnectionClosed, ConnectionError, OSError) as exc:
            if shutdown.is_set():
                break
            log.warning("Connection lost (%s) — reconnecting in %ds", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)

        except asyncio.CancelledError:
            break

    writer.close()
    log.info("Done. %d minute bars written to %s", minutes_written, output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="BTC 1-minute WebSocket collector for Coinbase Exchange"
    )
    parser.add_argument(
        "--product",
        default="BTC-USD",
        help="Coinbase product ID (default: BTC-USD)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUT),
        help=f"Output CSV path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=None,
        help="Stop after N minute bars (default: run indefinitely)",
    )
    args = parser.parse_args()

    asyncio.run(collect(
        product_id=args.product,
        output=Path(args.output),
        max_minutes=args.minutes,
    ))


if __name__ == "__main__":
    main()
