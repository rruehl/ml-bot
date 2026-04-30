"""
production_bot_v8_ml.py
=======================
Kalshi BTC Binary Options Trading Bot — Version 8.0.0
"ML Sniper — two_class_model replaces UT Bot signal engine"

Architecture changes from v7:
  - DELETED : CandleBuilder, ohlcv_candle_loop, calculate_ut_bot
  - DELETED : UT_BOT_SENSITIVITY, UT_BOT_ATR_PERIOD, CANDLE_TIMEFRAME params
  - DELETED : STOP_ATR_* thresholds used for ATR-bucketed Kelly / stop sizing
  - ADDED   : _MLInference — loads artifacts once at startup, runs predict_proba live
  - ADDED   : _calculate_live_atr_14() — live ATR gate matching train.py _compute_atr math
  - CHANGED : ML features now sourced from data/btc_1min.csv via BTCMinuteProcessor
  - ADDED   : SharedState.ml_* fields replace ut_signal / ut_atr / ut_stop
  - CHANGED : on_tick entry gate now checks ml_confidence >= ML_CONFIDENCE_TAU
  - CHANGED : RiskEngine.calculate_qty uses flat Kelly (no ATR bucket)
  - KEPT    : All Kalshi execution plumbing — WS, fill recovery, amend, stop engine,
              settlement, reconciliation, dashboard-compatible CSV log
"""

import asyncio
import collections
import csv
import hashlib
import json
import logging
import os
import sys
import time
import uuid
import websockets
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import ccxt.pro as ccxt
from dotenv import load_dotenv

from kalshi_client import KalshiClient

VERSION = "8.0.0 - MLSniper"

current_dir  = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(current_dir))
sys.path.append(str(project_root))
load_dotenv(dotenv_path=project_root / ".env")

STOP_DEBUG_LOG = str(project_root / "logs" / "stop_debug.log")
ESC_DEBUG_LOG  = str(project_root / "logs" / "escalation_debug.log")

# ── Artifact paths ────────────────────────────────────────────────────────────
ARTIFACTS_DIR   = project_root / "artifacts"
MODEL_PATH      = ARTIFACTS_DIR / "two_class_model.joblib"
SCALER_PATH     = ARTIFACTS_DIR / "scaler.joblib"
IMPUTER_PATH    = ARTIFACTS_DIR / "imputer.joblib"
SCHEMA_PATH     = ARTIFACTS_DIR / "feature_schema.json"
FLAG_PATH       = ARTIFACTS_DIR / "model_updated.flag"


def _setup_debug_logger(name: str, filepath: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    handler = RotatingFileHandler(filepath, maxBytes=5_000_000, backupCount=5)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


esc_log  = _setup_debug_logger("esc_debug_v8",  ESC_DEBUG_LOG)
stop_log = _setup_debug_logger("stop_debug_v8", STOP_DEBUG_LOG)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

class Config:
    PAPER_MODE          = True
    PAPER_START_BALANCE = 1000.0
    MAX_DAILY_LOSS      = 1_000_000.0
    STARTING_DEPOSIT    = 1000.0

    SYMBOL        = "BTC/USD"
    EXCHANGES     = ["coinbase"]
    SERIES_TICKER = "KXBTC15M"

    # ── ML signal parameters ──────────────────────────────────────────────────
    ML_CONFIDENCE_TAU    = 0.50    # Fallback default; overwritten at runtime from decision_threshold.json
    ML_SPREAD_MAX_CENTS     = 4       # Max Kalshi spread (¢) at inference time
    # Inference fires ONCE per session, within this many minutes of session open.
    # The model predicts the full 15-min window outcome; running it at minute 5
    # or 10 is out-of-distribution. Default: must fire within first 2 minutes.
    ML_INFERENCE_WINDOW_MIN = 2.0
    # Moneyness gate: max bps the Kalshi strike can be AGAINST our direction.
    # For YES: (Strike - Spot) / Spot * 10000. For NO: (Spot - Strike) / Spot * 10000.
    # Positive = strike disadvantageous. Reject if above this threshold.
    MAX_STRIKE_DISADVANTAGE_BPS = 20.0
    # ATR volatility gate: live ATR_14 (in USD) must fall within this band.
    # Mirrors the atr_min/atr_max channel in train.py.
    ATR_MIN = 15.0
    ATR_MAX = 30.0

    # ── Entry window ──────────────────────────────────────────────────────────
    TIME_ENTRY_MIN_MIN   = 15      # Don't enter if more than this many minutes left
    TIME_ENTRY_MAX_MIN   = 1       # Don't enter if fewer than this many minutes left
    MAKER_MAX_ENTRY_PRICE = 55     # Absolute price ceiling for our maker bid (¢)
    ENTRY_TTL_SECONDS    = 60      # Cancel resting order after this many seconds

    # ── Position sizing (flat Kelly — no ATR bucket) ──────────────────────────
    KELLY_FRACTION       = 0.25
    KELLY_WIN_RATE       = 0.571   # From metrics_two_class.json win_rate
    MAX_CONTRACTS_LIMIT  = 500
    MAX_FILLS_PER_SESSION = 1

    # ── Stop loss ─────────────────────────────────────────────────────────────
    # v8: hold-to-expiry is the primary exit; stop engine kept as safety net
    STOP_TRAIL_CENTS      = 25     # Flat trailing stop (¢) replaces ATR-bucketed trail
    STOP_FLOOR_CENTS      = 10
    STOP_TL_SH_THRESHOLD  = 4.8   # Short-time-left bucket cutoff (min)
    STOP_TL_LG_THRESHOLD  = 9.0   # Large-time-left bucket cutoff (min)
    STOP_DELAY_SH         = 8
    STOP_DELAY_MD         = 20
    STOP_DELAY_LG         = 35
    STOP_EXIT_MAX_RETRIES     = 4
    STOP_EXIT_RETRY_INCREMENT = 3
    STOP_RESCUE_INTERVAL_SEC  = 12.0
    STOP_CONFIRM_DELAY_SEC    = 5.0

    # ── Order management ──────────────────────────────────────────────────────
    REPRICE_THRESHOLD    = 2
    AMEND_COOLDOWN_SEC   = 4.0
    TAKER_FORCE_MIN_LEFT = 2.0
    TAKER_ENTRY_MIN_LEFT = 5.0

    LADDER_THRESHOLD     = 10
    LADDER_FILL_FRACTION = 0.7

    # ── Misc ──────────────────────────────────────────────────────────────────
    LOG_FILE        = str(project_root / "logs" / "production_log.csv")
    SYSTEM_LOG_FILE = str(project_root / "logs" / "bot.log")
    STATE_DIR       = str(current_dir / "state")
    STATE_FILE      = str(current_dir / "state" / "acted_birth_ts.json")

    HEARTBEAT_INTERVAL_SEC    = 10.0
    MAX_ORDERBOOK_STALE_SEC   = 10.0
    SETTLEMENT_INITIAL_DELAY  = 90.0
    SETTLEMENT_RETRY_INTERVAL = 15.0
    SETTLEMENT_MAX_RETRIES    = 4
    POST_ONLY_CANCEL_POLL_SEC = 3.0
    RECONCILE_DIVERGENCE_THRESHOLD = 0.10

    KALSHI_WS_URL     = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    LOG_ROTATE_MAX_MB = 50.0



_CONFIG_IMMUTABLE = frozenset({
    "PAPER_MODE", "PAPER_START_BALANCE", "SERIES_TICKER",
    "LOG_FILE", "STATE_FILE", "STATE_DIR", "SYSTEM_LOG_FILE",
})


def rotate_csv_log_if_needed():
    path = Path(Config.LOG_FILE)
    if not path.exists():
        return
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb < Config.LOG_ROTATE_MAX_MB:
        return
    stamp    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    new_name = path.with_suffix(f".{stamp}.csv")
    path.rename(new_name)
    with open(Config.LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(LOG_COLUMNS)
    print(f"[LOG] Rotated {path.name} → {new_name.name} ({size_mb:.1f}MB)")


def _config_file() -> str:
    cfg = str(current_dir / "config.json")
    return cfg


def load_config_at_startup():
    cfg_path = _config_file()
    if not os.path.exists(cfg_path):
        print("[CONFIG] WARNING: config.json not found — using class defaults.")
        return
    logger = logging.getLogger("kalshi_bot_v8")
    try:
        with open(cfg_path) as f:
            new = json.load(f)
    except Exception as e:
        print(f"[CONFIG] WARNING: Failed to parse {cfg_path} — using defaults. Error: {e}")
        logger.warning(f"{cfg_path} parse error: {e}")
        return

    for key, value in new.items():
        if not hasattr(Config, key):
            continue
        try:
            original = getattr(Config, key)
            if isinstance(original, bool):
                value = bool(value)
            elif isinstance(original, int):
                value = int(value)
            elif isinstance(original, float):
                value = float(value)
            setattr(Config, key, value)
        except (ValueError, TypeError) as e:
            print(f"[CONFIG] WARNING: Could not coerce {key}={value!r}: {e}")
            logger.warning(f"Config coerce error {key}: {e}")

    mode = "PAPER" if Config.PAPER_MODE else "*** LIVE ***"
    print(f"[CONFIG] Loaded — mode={mode}  tau={Config.ML_CONFIDENCE_TAU:.3f}  "
          f"deposit=${Config.STARTING_DEPOSIT:.2f}")


def update_config_from_file():
    cfg_path = _config_file()
    if not os.path.exists(cfg_path):
        return
    logger = logging.getLogger("kalshi_bot_v8")
    try:
        current_mtime = os.path.getmtime(cfg_path)
        if current_mtime <= update_config_from_file._last_mtime:
            return
    except OSError:
        return

    try:
        with open(cfg_path) as f:
            new = json.load(f)
    except Exception as e:
        print(f"[CONFIG] WARNING: Failed to parse {cfg_path}: {e}")
        logger.warning(f"{cfg_path} parse error: {e}")
        return

    update_config_from_file._last_mtime = current_mtime
    for key, value in new.items():
        if key in _CONFIG_IMMUTABLE or not hasattr(Config, key):
            continue
        try:
            original = getattr(Config, key)
            if isinstance(original, bool):
                value = bool(value)
            elif isinstance(original, int):
                value = int(value)
            elif isinstance(original, float):
                value = float(value)
            setattr(Config, key, value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Config coerce error {key}: {e}")

update_config_from_file._last_mtime = 0.0


def make_client_order_id(ticker: str, birth_ts: float, action: str, slice_idx: int = 0) -> str:
    raw = f"{ticker}|{birth_ts:.3f}|{action}|{slice_idx}|{int(time.time())}"
    return hashlib.md5(raw.encode()).hexdigest()


def setup_logging():
    logger = logging.getLogger("kalshi_bot_v8")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    handler = RotatingFileHandler(Config.SYSTEM_LOG_FILE, maxBytes=5_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    return logger


# ═════════════════════════════════════════════════════════════════════════════
# ML INFERENCE ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class _MLInference:
    """Loads model artifacts once at startup. Thread-safe for async use (no IO in predict)."""

    def __init__(self):
        self.model    = None
        self.scaler   = None
        self.imputer  = None
        self.features: list[str] = []
        self._loaded  = False

    def load(self):
        """Call once from main() before starting the event loop."""
        print(f"[ML] Loading artifacts from {ARTIFACTS_DIR} ...")
        try:
            self.model   = joblib.load(MODEL_PATH)
            self.scaler  = joblib.load(SCALER_PATH)
            self.imputer = joblib.load(IMPUTER_PATH)
            with open(SCHEMA_PATH) as f:
                self.features = json.load(f)["feature_names"]
            self._loaded = True
            print(f"[ML] Loaded OK — {len(self.features)} features | "
                  f"tau={Config.ML_CONFIDENCE_TAU:.3f}")
        except Exception as e:
            raise RuntimeError(f"[ML] FATAL: Could not load artifacts: {e}") from e

    @property
    def ready(self) -> bool:
        return self._loaded

    def predict(self, feature_df: pd.DataFrame) -> tuple[int, float, float]:
        """
        Returns (direction, confidence) where:
          direction  = 1 (UP / YES) or 0 (DOWN / NO)
          confidence = max(proba_up, proba_down)
        """
        X = self.imputer.transform(feature_df)
        X = self.scaler.transform(X)
        proba     = self.model.predict_proba(X)[0]       # [proba_down, proba_up]
        proba_up  = float(proba[1])
        direction = 1 if proba_up >= 0.5 else 0
        confidence = proba_up if direction == 1 else (1.0 - proba_up)
        return direction, confidence, proba_up


# Global inference engine — loaded once at startup
ML = _MLInference()


def _calculate_live_atr_14(df: pd.DataFrame) -> float:
    """
    Computes a 14-period EWM ATR from OHLC data, matching _compute_atr in train.py exactly.
    Operates on a single-asset DataFrame (no symbol grouping loop needed).
    """
    high       = df["high"].values.astype(np.float64)
    low        = df["low"].values.astype(np.float64)
    close      = df["close"].values.astype(np.float64)
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    return float(pd.Series(tr).ewm(span=14, adjust=False).mean().iloc[-1])


def _normalize_ob_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw order book absolute price levels into price-relative distances
    and per-level volume imbalances, matching the equivalent block in
    BTCMinuteProcessor.engineer_features() in ml/preprocess.py exactly.

    For each snapshot ("open", "close") and each of 50 levels:
      ask_distance_{i}_{snap} = (ask_price - mid) / mid        [fraction above mid]
      bid_distance_{i}_{snap} = (mid - bid_price) / mid        [fraction below mid]
      imbalance_{i}_{snap}    = (bid_size - ask_size) / total  [bounded -1..1]

    Raw absolute price and size columns are dropped so the DataFrame passed to
    engineer_features() contains only stationary, price-independent features.

    NOTE: Must be kept in sync with preprocess.py if the normalization logic changes.
    """
    _OB_LEVELS = 50
    cols_to_drop: list[str] = []

    for snap in ("open", "close"):
        bid_col = f"best_bid_{snap}"
        ask_col = f"best_ask_{snap}"
        if bid_col not in df.columns or ask_col not in df.columns:
            continue

        mid = (df[bid_col] + df[ask_col]) / 2.0
        mid = mid.replace(0, np.nan)

        for i in range(_OB_LEVELS):
            ask_p = f"ask[{i}].price_{snap}"
            bid_p = f"bid[{i}].price_{snap}"
            ask_s = f"ask[{i}].size_{snap}"
            bid_s = f"bid[{i}].size_{snap}"

            if ask_p in df.columns:
                df[f"ask_distance_{i}_{snap}"] = (df[ask_p] - mid) / mid
                cols_to_drop.append(ask_p)

            if bid_p in df.columns:
                df[f"bid_distance_{i}_{snap}"] = (mid - df[bid_p]) / mid
                cols_to_drop.append(bid_p)

            ask_s_ok = ask_s in df.columns
            bid_s_ok = bid_s in df.columns
            if ask_s_ok and bid_s_ok:
                total_vol = df[bid_s] + df[ask_s]
                df[f"imbalance_{i}_{snap}"] = (
                    (df[bid_s] - df[ask_s]) / total_vol.replace(0, 1)
                )
                cols_to_drop.extend([ask_s, bid_s])
            else:
                if ask_s_ok:
                    cols_to_drop.append(ask_s)
                if bid_s_ok:
                    cols_to_drop.append(bid_s)

    if cols_to_drop:
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df


# ═════════════════════════════════════════════════════════════════════════════
# RISK ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class RiskEngine:
    def __init__(self, start_balance: float):
        self.paper_balance = start_balance
        self.real_balance  = 0.0
        self.pnl_history   = deque(maxlen=5000)

    def record_pnl(self, amount: float):
        self.pnl_history.append((time.time(), amount))

    def rolling_24h_loss(self) -> float:
        cutoff = time.time() - 86400
        return sum(pnl for ts, pnl in self.pnl_history if ts > cutoff and pnl < 0)

    async def sync_live_balance(self, kalshi, bot_ref=None):
        if Config.PAPER_MODE:
            return
        try:
            resp = await kalshi.get_balance()
            if "balance" in resp:
                self.real_balance = resp["balance"] / 100.0
            print(f"[RISK] Balance synced: ${self.real_balance:.2f}")
            if bot_ref is not None:
                bot_ref.log("BALANCE_SYNC", {}, f"Kalshi confirmed balance: ${self.real_balance:.2f}")
        except Exception as e:
            print(f"[RISK] WARNING: Failed to sync live balance: {e}")

    def calculate_qty(self, entry_price_cents: int) -> int:
        """Flat Kelly — no ATR bucket, uses model win rate from metrics."""
        bankroll = self.paper_balance if Config.PAPER_MODE else self.real_balance
        if abs(self.rolling_24h_loss()) > Config.MAX_DAILY_LOSS or bankroll <= 0:
            return 0
        p   = Config.KELLY_WIN_RATE
        q   = 1.0 - p
        b   = (100.0 - entry_price_cents) / entry_price_cents   # net odds
        kelly_full = (b * p - q) / b if b > 0 else 0.0
        kelly_full = max(0.0, kelly_full)
        dollar_risk = kelly_full * Config.KELLY_FRACTION * bankroll
        qty = int(dollar_risk / (Config.STOP_TRAIL_CENTS / 100.0))
        return min(max(0, qty), Config.MAX_CONTRACTS_LIMIT)


# ═════════════════════════════════════════════════════════════════════════════
# SHARED STATE  (ML flavour)
# ═════════════════════════════════════════════════════════════════════════════

class SharedState:
    """
    Stores the live BTC spot price (from the CCXT ticker stream) for the
    moneyness gate and heartbeat logs, plus the per-session ML inference result.
    """

    def __init__(self):
        self.latest_btc: float = 0.0

        # ML inference result — locked in ONCE per session at session open.
        # ml_session_fired prevents re-running predict_proba after the first call.
        self.ml_direction    = None   # 1 = UP / YES,  0 = DOWN / NO,  None = not yet fired
        self.ml_confidence   = 0.0   # max(proba_up, proba_down)
        self.ml_proba_up     = 0.0   # raw model output
        self.ml_birth_ts     = 0.0   # wall-clock time inference fired
        self.ml_session_fired = False # True once predict_proba has run this session

    def update_ml(self, direction: int, confidence: float, proba_up: float):
        self.ml_direction    = direction
        self.ml_confidence   = confidence
        self.ml_proba_up     = proba_up
        self.ml_birth_ts     = time.time()
        self.ml_session_fired = True

    def clear_session_ml(self):
        """Call on every session roll to reset the per-session inference slot."""
        self.ml_direction    = None
        self.ml_confidence   = 0.0
        self.ml_proba_up     = 0.0
        self.ml_birth_ts     = 0.0
        self.ml_session_fired = False


# ═════════════════════════════════════════════════════════════════════════════
# LIVE ORDERBOOK  (unchanged from v7)
# ═════════════════════════════════════════════════════════════════════════════

class LiveOrderbook:
    RESYNC_INTERVAL_SEC = 30.0

    def __init__(self):
        self.lock           = asyncio.Lock()
        self.ticker         = None
        self.strike         = 0.0
        self.close_time     = None
        self.yes_levels: dict = {}
        self.no_levels:  dict = {}
        self.last_update_ts = 0.0
        self.last_delta_ts  = 0.0
        self.last_resync_ts = 0.0
        self.ws_connected   = False
        self.ticker_changed = asyncio.Event()

    def needs_resync(self) -> bool:
        return (time.time() - self.last_resync_ts) > self.RESYNC_INTERVAL_SEC

    async def apply_rest_orderbook(self, ob_resp: dict):
        ob_fp    = ob_resp.get("orderbook_fp", {})
        yes_data = ob_fp.get("yes_dollars", [])
        no_data  = ob_fp.get("no_dollars",  [])
        async with self.lock:
            self.yes_levels = {round(float(p) * 100): float(q)
                               for p, q in yes_data
                               if 1 <= round(float(p) * 100) <= 99}
            self.no_levels  = {round(float(p) * 100): float(q)
                               for p, q in no_data
                               if 1 <= round(float(p) * 100) <= 99}
            self.last_update_ts = time.time()
            self.last_delta_ts  = time.time()
            self.last_resync_ts = time.time()

    async def reset(self, ticker: str, strike: float, close_time: str):
        async with self.lock:
            self.ticker         = ticker
            self.strike         = strike
            self.close_time     = close_time
            self.yes_levels     = {}
            self.no_levels      = {}
            self.last_update_ts = time.time()
            self.last_resync_ts = 0.0
        self.ticker_changed.set()

    async def apply_snapshot(self, yes_fp: list, no_fp: list):
        async with self.lock:
            self.yes_levels = {round(float(p) * 100): float(q)
                               for p, q in yes_fp
                               if 1 <= round(float(p) * 100) <= 99}
            self.no_levels  = {round(float(p) * 100): float(q)
                               for p, q in no_fp
                               if 1 <= round(float(p) * 100) <= 99}
            self.last_update_ts = time.time()
            self.last_delta_ts  = time.time()
            self.last_resync_ts = time.time()

    async def apply_delta(self, side: str, price_cents: int, delta_qty: float):
        book = self.yes_levels if side == "yes" else self.no_levels
        if not (1 <= price_cents <= 99):
            return
        new_qty = book.get(price_cents, 0.0) + delta_qty
        if new_qty <= 0:
            book.pop(price_cents, None)
        else:
            book[price_cents] = new_qty
        self.last_update_ts = time.time()
        self.last_delta_ts  = time.time()

    def snapshot(self) -> dict | None:
        if not self.ticker:
            return None
        if not self.yes_levels and not self.no_levels:
            return None

        now = datetime.now(timezone.utc)
        if self.close_time:
            close_dt     = datetime.fromisoformat(self.close_time.replace("Z", "+00:00"))
            minutes_left = max(0.0, (close_dt - now).total_seconds() / 60.0)
        else:
            minutes_left = 0.0

        y_bid  = max(self.yes_levels.keys(), default=0)
        n_bid  = max(self.no_levels.keys(),  default=0)
        y_top5 = sorted(self.yes_levels.items(), reverse=True)[:5]
        n_top5 = sorted(self.no_levels.items(),  reverse=True)[:5]
        y_liq  = sum(q for _, q in y_top5)
        n_liq  = sum(q for _, q in n_top5)

        return {
            "ticker":       self.ticker,
            "strike":       self.strike,
            "minutes_left": minutes_left,
            "raw_yes_bid":  y_bid,
            "raw_no_bid":   n_bid,
            "ask_yes":      100 - n_bid if n_bid > 0 else 99,
            "ask_no":       100 - y_bid if y_bid > 0 else 99,
            "yes_liq":      y_liq,
            "no_liq":       n_liq,
            "yes_depth":    sorted(self.yes_levels.items(), reverse=True),
            "no_depth":     sorted(self.no_levels.items(),  reverse=True),
            "obi":          (y_liq - n_liq) / (y_liq + n_liq) if (y_liq + n_liq) > 0 else 0.0,
        }


# ═════════════════════════════════════════════════════════════════════════════
# LOG COLUMNS  (v8 replaces ut_signal/ut_atr/ut_stop with ml_* fields)
# ═════════════════════════════════════════════════════════════════════════════

LOG_COLUMNS = [
    "timestamp", "event", "mode",
    "ticker", "side", "entry_price", "qty",
    "time_left", "btc_price", "strike",
    "raw_yes_bid", "raw_no_bid", "ask_yes", "ask_no", "spread",
    "yes_liq", "no_liq", "obi",
    "bankroll", "rolling_24h_loss",
    # ML replaces ut_signal / ut_atr / ut_stop
    "ml_direction", "ml_confidence", "ml_proba_up", "ml_tau",
    "ml_birth_ts", "signal_age_min",
    "ob_stale", "ob_last_delta_age", "ws_connected",
    "filter_reason",
    "order_id", "client_order_id", "order_status",
    "kalshi_fill_qty", "kalshi_fill_price",
    "kalshi_fill_cost", "kalshi_fees",
    "stop_trail", "stop_best_bid", "stop_active",
    "settlement_source", "btc_price_at_settlement",
    "gross_proceeds", "gross_cost", "net_fees", "pnl_this_trade",
    "reconcile_estimated_pnl", "reconcile_actual_pnl", "reconcile_delta",
    "msg"
]


# ═════════════════════════════════════════════════════════════════════════════
# FILL COST HELPERS  (unchanged from v7)
# ═════════════════════════════════════════════════════════════════════════════

def _parse_fill_cost_and_fees(fills: list, action_filter: str) -> tuple[float, float, float, float]:
    total_cost = total_fees = total_qty = total_value = 0.0
    for f in fills:
        if f.get("action") != action_filter:
            continue
        qty  = float(f.get("count_fp") or 0)
        fee  = float(f.get("fee_cost") or 0)
        side = f.get("side", "")
        if side == "no":
            raw_yes = float(f.get("yes_price_dollars") or 0)
            price = float(f.get("no_price_dollars") or (1.0 - raw_yes if raw_yes > 0 else 0))
        else:
            price = float(f.get("yes_price_dollars") or f.get("no_price_dollars") or 0)
        cost         = qty * price
        total_cost  += cost
        total_fees  += fee
        total_qty   += qty
        total_value += qty * price
    avg_price_cents = round((total_value / total_qty) * 100) if total_qty > 0 else 0
    return total_cost, total_fees, total_qty, avg_price_cents


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY CONTROLLER
# ═════════════════════════════════════════════════════════════════════════════

class StrategyController:

    def __init__(self, shared_state: SharedState, live_ob: LiveOrderbook):
        self.risk   = RiskEngine(Config.PAPER_START_BALANCE)
        self.shared = shared_state
        self.ob     = live_ob

        self.active_position = None
        self.active_order    = None

        self.session_fills      = 0
        self.prev_ticker        = None
        self.prev_strike        = 0.0
        self.session_start_time = time.time()

        # Stop engine state
        self._stop_best_bid         = 0
        self._stop_tick_count       = 0
        self._stop_trail            = 0
        self._stop_delay_sec        = 0.0
        self._stop_entry_time       = 0.0
        self._stop_active           = False
        self._stop_exit_in_progress = False
        self._stop_failed_ticker: str = ""
        self._stop_exit_submitted: bool = False
        self._rescue_in_progress: bool = False

        self.last_heartbeat_ts = 0.0
        self._settle_lock      = asyncio.Lock()
        self.circuit_breaker_open    = False
        self._circuit_breaker_logged = False
        self._kalshi_ref             = None
        self._last_recovered_order_id: str = ""

        # ML inference session guard
        self.acted_on_ml_birth_ts = self._load_birth_time()

        Path(Config.STATE_DIR).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(Config.LOG_FILE):
            with open(Config.LOG_FILE, "w", newline="") as f:
                csv.writer(f).writerow(LOG_COLUMNS)

    # ── State persistence ────────────────────────────────────────────────────

    def _load_birth_time(self):
        try:
            if os.path.exists(Config.STATE_FILE):
                with open(Config.STATE_FILE) as f:
                    return json.load(f).get("acted_on_birth_time")
        except Exception:
            pass
        return None

    def _save_birth_time(self, birth_ts):
        try:
            with open(Config.STATE_FILE, "w") as f:
                json.dump({"acted_on_birth_time": birth_ts}, f)
        except Exception:
            pass

    # ── Fill handling ─────────────────────────────────────────────────────────

    async def on_fill(self, fill_msg: dict):
        if self.active_order is None:
            return
        order_id = fill_msg.get("order_id", "")
        if order_id != self.active_order.get("order_id"):
            return
        fill_qty = float(fill_msg.get("count_fp", fill_msg.get("count", 0)))
        if fill_qty <= 0:
            return

        ao          = self.active_order
        traded_side = ao["side"]
        yes_price_raw = float(fill_msg.get("yes_price_dollars") or 0)
        if traded_side == "yes":
            fill_price = round(yes_price_raw * 100)
        else:
            no_price_raw = float(fill_msg.get("no_price_dollars") or (1.0 - yes_price_raw if yes_price_raw > 0 else 0))
            fill_price = round(no_price_raw * 100)

        data         = self.ob.snapshot() or {}
        actual_entry = fill_price if fill_price > 0 else ao["posted_price"]

        self.active_position = {
            "ticker":          ao["ticker"],
            "side":            traded_side,
            "qty":             ao["qty"],
            "entry_price":     actual_entry,
            "order_id":        ao["order_id"],
            "client_order_id": ao["client_order_id"],
            "ml_direction":    ao.get("ml_direction"),
            "ml_confidence":   ao.get("ml_confidence", 0.0),
            "ml_proba_up":     ao.get("ml_proba_up", 0.0),
            "ml_birth_ts":     ao["birth_ts"],
            "signal_age_min":  ao.get("signal_age_min", 0.0),
            "entry_cost":      fill_qty * actual_entry / 100.0,
            "entry_fees":      0.0,
        }
        self.active_order = None
        self._init_stop(actual_entry, data.get("minutes_left", 0.0))

        posted_price = ao.get("posted_price", actual_entry)
        fill_ctx = {
            "ticker":          ao["ticker"],
            "side":            traded_side,
            "entry_price":     posted_price,
            "qty":             int(fill_qty),
            "time_left":       data.get("minutes_left", 0.0),
            "btc":             self.shared.latest_btc,
            "strike":          data.get("strike", 0),
            "raw_yes_bid":     data.get("raw_yes_bid", 0),
            "raw_no_bid":      data.get("raw_no_bid", 0),
            "ask_yes":         data.get("ask_yes", 0),
            "ask_no":          data.get("ask_no", 0),
            "yes_liq":         data.get("yes_liq", 0),
            "no_liq":          data.get("no_liq", 0),
            "obi":             data.get("obi", 0.0),
            "order_id":        ao["order_id"],
            "client_order_id": ao["client_order_id"],
            "order_status":    "filled",
            "kalshi_fill_qty":   fill_qty,
            "kalshi_fill_price": actual_entry,
            "kalshi_fill_cost":  0.0,
            "kalshi_fees":       0.0,
            "ml_direction":    ao.get("ml_direction"),
            "ml_confidence":   ao.get("ml_confidence", 0.0),
            "ml_proba_up":     ao.get("ml_proba_up", 0.0),
            "ml_birth_ts":     ao["birth_ts"],
            "signal_age_min":  ao.get("signal_age_min", 0.0),
        }
        self.log("FILL_CONFIRMED", fill_ctx,
                 f"Kalshi fill: {traded_side.upper()} {fill_qty:.0f}x @ {actual_entry}¢ "
                 f"(posted={posted_price}¢, slippage={actual_entry - posted_price:+d}¢)")
        print(f"\033[92m[FILL CONFIRMED] {ao['ticker']} | {traded_side.upper()} @ "
              f"{actual_entry}¢ x{fill_qty:.0f}\033[0m")
        asyncio.create_task(self._fetch_fill_costs(ao["order_id"], ao["ticker"]))

    async def _fetch_fill_costs(self, order_id: str, ticker: str):
        await asyncio.sleep(2.0)
        if Config.PAPER_MODE:
            return
        try:
            if not hasattr(self, '_kalshi_ref') or self._kalshi_ref is None:
                return
            kalshi     = self._kalshi_ref
            fills_resp = await kalshi.get_fills(order_id=order_id)
            fills      = fills_resp.get("fills", [])
            entry_cost, entry_fees, fill_qty, avg_price = _parse_fill_cost_and_fees(fills, "buy")
            if fill_qty > 0:
                is_taker = any(f.get("is_taker") for f in fills)
                if is_taker:
                    recomputed = round(entry_cost * 0.02, 6)
                    if recomputed > entry_fees + 1e-9:
                        entry_fees = recomputed
            if fill_qty > 0 and self.active_position is not None:
                self.active_position["entry_cost"] = entry_cost
                self.active_position["entry_fees"] = entry_fees
            self.log("FILL_VERIFIED", {
                "ticker": ticker, "order_id": order_id,
                "kalshi_fill_qty":   fill_qty,
                "kalshi_fill_price": avg_price,
                "kalshi_fill_cost":  entry_cost,
                "kalshi_fees":       entry_fees,
            }, f"Fill cost verified: qty={fill_qty:.0f} cost=${entry_cost:.4f} fees=${entry_fees:.6f}")
            print(f"\033[92m[FILL VERIFIED] {ticker} | cost=${entry_cost:.4f} fees=${entry_fees:.6f}\033[0m")
        except Exception as e:
            print(f"[FILL] WARNING: Could not fetch fill costs for {order_id}: {e}")

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, event, ctx, msg=""):
        ts   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        bank = self.risk.paper_balance if Config.PAPER_MODE else self.risk.real_balance
        mode = "PAPER" if Config.PAPER_MODE else "LIVE"

        yes_bid = ctx.get("raw_yes_bid", 0)
        no_bid  = ctx.get("raw_no_bid", 0)
        ask_yes = ctx.get("ask_yes", 0)
        ask_no  = ctx.get("ask_no", 0)
        side    = ctx.get("side", "")

        if side == "yes":
            spread = ask_yes - yes_bid if yes_bid > 0 else 0
        elif side == "no":
            spread = ask_no - no_bid if no_bid > 0 else 0
        else:
            spread = ask_yes - yes_bid if yes_bid > 0 else 0

        ob_stale          = int((time.time() - self.ob.last_update_ts) > Config.MAX_ORDERBOOK_STALE_SEC)
        ob_last_delta_age = round(time.time() - self.ob.last_delta_ts, 1) if self.ob.last_delta_ts > 0 else 9999.0
        ws_connected      = 1 if self.ob.ws_connected else 0

        row = [
            ts, event, mode,
            ctx.get("ticker", ""), side, ctx.get("entry_price", 0), ctx.get("qty", 0),
            round(ctx.get("time_left", 0.0), 2), round(ctx.get("btc", 0.0), 2), ctx.get("strike", 0),
            yes_bid, no_bid, ask_yes, ask_no, spread,
            ctx.get("yes_liq", 0), ctx.get("no_liq", 0), round(ctx.get("obi", 0.0), 3),
            round(bank, 2), round(self.risk.rolling_24h_loss(), 2),
            ctx.get("ml_direction", ""), round(ctx.get("ml_confidence", 0.0), 4),
            round(ctx.get("ml_proba_up", 0.0), 4),
            round(ctx.get("ml_tau", 0.0), 6),
            ctx.get("ml_birth_ts", 0), round(ctx.get("signal_age_min", 0.0), 2),
            ob_stale, ob_last_delta_age, ws_connected,
            ctx.get("filter_reason", ""),
            ctx.get("order_id", ""), ctx.get("client_order_id", ""), ctx.get("order_status", ""),
            round(ctx.get("kalshi_fill_qty", 0), 2),
            ctx.get("kalshi_fill_price", 0),
            round(ctx.get("kalshi_fill_cost", 0.0), 6),
            round(ctx.get("kalshi_fees", 0.0), 6),
            self._stop_trail, self._stop_best_bid, 1 if self._stop_active else 0,
            ctx.get("settlement_source", ""),
            round(ctx.get("btc_price_at_settlement", 0.0), 2),
            round(ctx.get("gross_proceeds", 0.0), 6),
            round(ctx.get("gross_cost", 0.0), 6),
            round(ctx.get("net_fees", 0.0), 6),
            round(ctx.get("pnl_this_trade", 0.0), 6),
            round(ctx.get("reconcile_estimated_pnl", 0.0), 6),
            round(ctx.get("reconcile_actual_pnl", 0.0), 6),
            round(ctx.get("reconcile_delta", 0.0), 6),
            msg
        ]

        try:
            with open(Config.LOG_FILE, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as _csv_exc:
            _fb = logging.getLogger("kalshi_bot_v8")
            _fb.error("CSV write failed event=%s error=%s row=%s", event, _csv_exc, row)

        try:
            if event == "HRTBT":
                dir_str  = {1: "UP", 0: "DN", None: "--"}.get(ctx.get("ml_direction"), "--")
                conf_str = f"{ctx.get('ml_confidence', 0.0):.3f}"
                stale    = "  [STALE-OB]" if ob_stale else ""
                nows     = "  [NOT CONNECTED]" if not ws_connected else ""
                in_pos   = "  [IN POSITION]"  if ctx.get("has_position") else ""
                resting  = "  [ORDER RESTING]" if ctx.get("has_order") else ""
                print(
                    f"\033[90m[{'HRTBT':^10}] "
                    f"{ctx.get('ticker','N/A')} | BTC:{ctx.get('btc',0):.2f} | "
                    f"ML:{dir_str}@{conf_str} | "
                    f"Tau:{Config.ML_CONFIDENCE_TAU:.3f} | ATR:[{Config.ATR_MIN}-{Config.ATR_MAX}] | "
                    f"Y:{yes_bid}c N:{no_bid}c OBI:{ctx.get('obi',0):+.3f} | "
                    f"Trail:{self._stop_trail}¢ Best:{self._stop_best_bid}¢ Active:{self._stop_active} | "
                    f"Bank:${bank:.2f} ΔAge:{ob_last_delta_age:.0f}s"
                    f"{stale}{nows}{in_pos}{resting}\033[0m"
                )
            elif event in ("FILL_CONFIRMED", "PAPER_BUY", "LIVE_BUY", "PAYOUT",
                           "ORDER_RESTING", "ORDER_AMENDED", "BALANCE_SYNC", "FILL_VERIFIED"):
                print(f"\033[92m[{event:^18}] {ctx.get('ticker','')} | {msg}\033[0m")
            elif event in ("STOP_EXIT", "STOP_CONFIRMED", "STOP_FAILED", "STOP_RETRY",
                           "ORDER_UNFILLED", "SETTLE", "ERROR"):
                print(f"\033[91m[{event:^18}] {ctx.get('ticker','')} | {msg}\033[0m")
            elif event == "SETTLE_VERIFIED":
                print(f"\033[94m[{event:^18}] {ctx.get('ticker','')} | {msg}\033[0m")
            elif event == "ORDER_ESCALATED":
                print(f"\033[93m[{event:^18}] {ctx.get('ticker','')} | {msg}\033[0m")
            elif event == "ML_INFERENCE":
                print(f"\033[96m[{'ML_INFERENCE':^18}] {ctx.get('ticker','')} | {msg}\033[0m")
            else:
                print(f"[{event:^18}] {ctx.get('ticker','')} | {msg}")
        except OSError:
            pass

    # ── Stop engine (flat trail — no ATR bucket) ──────────────────────────────

    def _init_stop(self, entry_price_cents: int, time_left_min: float):
        self._stop_trail = Config.STOP_TRAIL_CENTS
        delay_ticks = (Config.STOP_DELAY_SH if time_left_min < Config.STOP_TL_SH_THRESHOLD
                       else Config.STOP_DELAY_LG if time_left_min >= Config.STOP_TL_LG_THRESHOLD
                       else Config.STOP_DELAY_MD)
        self._stop_delay_sec  = delay_ticks * 10.0
        self._stop_entry_time = time.time()
        self._stop_best_bid   = entry_price_cents
        self._stop_tick_count = 0
        self._stop_active     = (self._stop_delay_sec == 0.0)

    def _check_stop(self, current_bid: int) -> bool:
        if not self._stop_active:
            if time.time() - self._stop_entry_time >= self._stop_delay_sec:
                self._stop_active = True
            else:
                return False
        if current_bid > self._stop_best_bid:
            self._stop_best_bid = current_bid
        stop_price = self._stop_best_bid - self._stop_trail
        return current_bid <= max(stop_price, Config.STOP_FLOOR_CENTS)

    # ── Settlement ────────────────────────────────────────────────────────────

    async def _bg_settle(self, kalshi, ticker: str, btc_price: float,
                         strike: float, position: dict | None, order_id: str | None = None):
        async with self._settle_lock:
            await asyncio.sleep(Config.SETTLEMENT_INITIAL_DELAY)
            logger = setup_logging()
            if position is None:
                # No position — just log the outcome for dashboard context
                for _ in range(Config.SETTLEMENT_MAX_RETRIES + 1):
                    try:
                        md = (await kalshi.get_market(ticker)).get("market", {})
                        if md.get("status") in ("settled", "finalized") and md.get("result"):
                            result = md["result"].lower()
                            self.log("SETTLE_VERIFIED", {
                                "ticker": ticker,
                                "settlement_source": "kalshi_api_no_position",
                                "btc_price_at_settlement": btc_price,
                            }, f"Market settled {result.upper()} | no position held")
                            return
                        await asyncio.sleep(Config.SETTLEMENT_RETRY_INTERVAL)
                    except Exception:
                        await asyncio.sleep(Config.SETTLEMENT_RETRY_INTERVAL)
                self.log("SETTLE_VERIFIED", {
                    "ticker": ticker, "settlement_source": "spot_fallback_no_position",
                    "btc_price_at_settlement": btc_price,
                }, f"Market Roll: no position | spot fallback")
                return

            # Has position — determine outcome
            side = position["side"]
            qty  = position["qty"]

            source = "spot_fallback"
            verified = None
            for _ in range(Config.SETTLEMENT_MAX_RETRIES + 1):
                try:
                    md = (await kalshi.get_market(ticker)).get("market", {})
                    if md.get("status") in ("settled", "finalized") and md.get("result"):
                        verified = md["result"].lower()
                        source   = "kalshi_api"
                        break
                    await asyncio.sleep(Config.SETTLEMENT_RETRY_INTERVAL)
                except Exception:
                    await asyncio.sleep(Config.SETTLEMENT_RETRY_INTERVAL)

            if verified is None:
                verified = "yes" if btc_price > strike else "no"

            won            = (side == "yes" and verified == "yes") or (side == "no" and verified == "no")
            gross_proceeds = float(qty) * 1.00 if won else 0.0
            gross_cost     = position.get("entry_cost", position["entry_price"] * qty / 100.0)
            net_fees       = position.get("entry_fees", 0.0)
            pnl            = gross_proceeds - gross_cost - net_fees

            if Config.PAPER_MODE:
                self.risk.paper_balance += gross_proceeds
            self.risk.record_pnl(pnl)

            settle_ctx = {
                "ticker":                    ticker,
                "side":                      side,
                "entry_price":               position["entry_price"],
                "qty":                       qty,
                "btc":                       btc_price,
                "strike":                    strike,
                "settlement_source":         source,
                "btc_price_at_settlement":   btc_price,
                "order_id":                  position.get("order_id", ""),
                "ml_direction":              position.get("ml_direction"),
                "ml_confidence":             position.get("ml_confidence", 0.0),
                "ml_proba_up":               position.get("ml_proba_up", 0.0),
                "ml_birth_ts":               position.get("ml_birth_ts", 0),
                "signal_age_min":            position.get("signal_age_min", 0.0),
                "gross_proceeds":            gross_proceeds,
                "gross_cost":                gross_cost,
                "net_fees":                  net_fees,
                "pnl_this_trade":            pnl,
            }
            event = "PAYOUT" if won else "SETTLE"
            self.log(event, settle_ctx,
                     f"{'WIN' if won else 'LOSS'} | proceeds=${gross_proceeds:.4f} "
                     f"cost=${gross_cost:.4f} fees=${net_fees:.6f} pnl=${pnl:.4f}")
            await self.risk.sync_live_balance(kalshi, bot_ref=self)

            if not Config.PAPER_MODE:
                asyncio.create_task(self._reconcile_settlement(
                    kalshi, ticker, position, pnl, gross_proceeds, gross_cost, net_fees,
                    source, btc_price, strike
                ))

    async def _reconcile_settlement(
        self, kalshi, ticker: str, position: dict,
        estimated_pnl: float, est_proceeds: float, est_cost: float, est_fees: float,
        source: str, settlement_btc_price: float, strike: float,
    ):
        await asyncio.sleep(Config.SETTLEMENT_INITIAL_DELAY + 10.0)
        order_id = position.get("order_id", "")
        side     = position["side"]
        qty      = position["qty"]
        logger   = setup_logging()
        try:
            actual_cost = est_cost
            actual_entry_fees = est_fees
            if order_id:
                try:
                    fills_resp = await kalshi.get_fills(order_id=order_id)
                    fills      = fills_resp.get("fills", [])
                    buy_cost, buy_fees, buy_qty, _ = _parse_fill_cost_and_fees(fills, "buy")
                    if buy_qty > 0:
                        actual_cost       = buy_cost
                        actual_entry_fees = buy_fees
                except Exception as e:
                    logger.warning(f"[RECONCILE] Could not fetch entry fills for {order_id}: {e}")

            actual_proceeds  = 0.0
            settlement_result = None
            for _ in range(Config.SETTLEMENT_MAX_RETRIES + 2):
                try:
                    md = (await kalshi.get_market(ticker)).get("market", {})
                    if md.get("status") in ("settled", "finalized") and md.get("result"):
                        settlement_result = md["result"].lower()
                        break
                    await asyncio.sleep(Config.SETTLEMENT_RETRY_INTERVAL)
                except Exception:
                    await asyncio.sleep(Config.SETTLEMENT_RETRY_INTERVAL)

            if settlement_result is not None:
                won = ((side == "yes" and settlement_result == "yes") or
                       (side == "no"  and settlement_result == "no"))
                actual_proceeds = float(qty) if won else 0.0
            else:
                won = ((side == "yes" and settlement_btc_price > strike) or
                       (side == "no"  and settlement_btc_price <= strike))
                actual_proceeds = float(qty) if won else 0.0

            actual_pnl = actual_proceeds - actual_cost - actual_entry_fees
            delta      = actual_pnl - estimated_pnl
            diverged   = abs(delta) >= Config.RECONCILE_DIVERGENCE_THRESHOLD

            reconcile_ctx = {
                "ticker":                  ticker, "side": side,
                "entry_price":             position["entry_price"], "qty": qty,
                "settlement_source":       source,
                "btc_price_at_settlement": settlement_btc_price,
                "order_id":                order_id,
                "gross_proceeds":          actual_proceeds,
                "gross_cost":              actual_cost,
                "net_fees":                actual_entry_fees,
                "pnl_this_trade":          actual_pnl,
                "reconcile_estimated_pnl": estimated_pnl,
                "reconcile_actual_pnl":    actual_pnl,
                "reconcile_delta":         delta,
            }
            flag = "⚠️ DIVERGENCE" if diverged else "OK"
            self.log("RECONCILE", reconcile_ctx,
                     f"{flag} | estimated=${estimated_pnl:+.4f} actual=${actual_pnl:+.4f} "
                     f"delta=${delta:+.4f}")
            if diverged:
                print(f"\033[93m[RECONCILE] ⚠️  {ticker} | delta=${delta:+.4f}\033[0m")
                logger.warning(f"RECONCILE DIVERGENCE {ticker}: estimated={estimated_pnl:+.4f} "
                               f"actual={actual_pnl:+.4f} delta={delta:+.4f}")
            else:
                print(f"\033[94m[RECONCILE] OK  {ticker} | pnl=${actual_pnl:+.4f}\033[0m")
        except Exception as e:
            logger.error(f"[RECONCILE] Failed for {ticker}: {e}")

    # ── Stop exit (unchanged from v7) ─────────────────────────────────────────

    async def _stop_exit(self, kalshi, data):
        pos   = self.active_position
        side  = pos["side"]
        qty   = pos["qty"]
        entry = pos["entry_price"]

        stop_ctx = {
            "ticker": data["ticker"], "side": side, "entry_price": entry, "qty": qty,
            "time_left": data.get("minutes_left", 0.0), "strike": data["strike"],
            "btc": self.shared.latest_btc,
            "raw_yes_bid": data.get("raw_yes_bid", 0), "raw_no_bid": data.get("raw_no_bid", 0),
            "ask_yes": data.get("ask_yes", 0), "ask_no": data.get("ask_no", 0),
            "yes_liq": data.get("yes_liq", 0), "no_liq": data.get("no_liq", 0),
            "obi": data.get("obi", 0.0),
            "order_id": pos.get("order_id", ""), "client_order_id": pos.get("client_order_id", ""),
            "settlement_source": "trailing_stop",
        }

        if Config.PAPER_MODE:
            current_bid    = data["raw_yes_bid"] if side == "yes" else data["raw_no_bid"]
            exit_price     = max(current_bid, Config.STOP_FLOOR_CENTS)
            gross_proceeds = exit_price * qty / 100.0
            gross_cost     = pos.get("entry_cost", entry * qty / 100.0)
            pnl = gross_proceeds - gross_cost
            self.risk.paper_balance += gross_proceeds
            self.risk.record_pnl(pnl)
            self.active_position = None
            self.session_fills   = 0
            self._stop_exit_in_progress = False
            self.log("STOP_EXIT", {
                **stop_ctx, "gross_proceeds": gross_proceeds,
                "gross_cost": gross_cost, "pnl_this_trade": pnl,
            }, f"PAPER stop: {side.upper()} exit @ {exit_price}¢ | pnl=${pnl:.4f}")
            return

        # Live stop — IoC ladder with rescue
        price_floor = max(
            max(data["raw_yes_bid"] if side == "yes" else data["raw_no_bid"], 1),
            Config.STOP_FLOOR_CENTS
        )
        stop_submitted = False
        exit_order_id  = ""
        attempt_ctx    = dict(stop_ctx)

        for attempt in range(Config.STOP_EXIT_MAX_RETRIES + 1):
            try:
                current_bid = data["raw_yes_bid"] if side == "yes" else data["raw_no_bid"]
                price_floor = max(current_bid - attempt * Config.STOP_EXIT_RETRY_INCREMENT,
                                  Config.STOP_FLOOR_CENTS)
                coid = make_client_order_id(data["ticker"], pos.get("ml_birth_ts", 0), f"stop_{attempt}")
                order_kwargs = {
                    "ticker": data["ticker"], "action": "sell", "side": side, "count": qty,
                    "time_in_force": "immediate_or_cancel", "client_order_id": coid,
                }
                if side == "yes":
                    order_kwargs["yes_price"] = price_floor
                else:
                    order_kwargs["no_price"] = price_floor

                resp          = await kalshi.create_order(**order_kwargs)
                exit_order_id = resp.get("order", {}).get("order_id", "")
                stop_submitted = True
                self._stop_exit_submitted = True
                self.log("STOP_EXIT", {**attempt_ctx, "order_id": exit_order_id},
                         f"IoC stop attempt {attempt+1}: {side.upper()} sell @ {price_floor}¢ x{qty}")

                ioc_filled = False
                if not Config.PAPER_MODE and exit_order_id:
                    await asyncio.sleep(1.0)
                    try:
                        order_resp = await kalshi.get_order(exit_order_id)
                        fill_count = float(order_resp.get("order", {}).get("fill_count_fp", "0"))
                        ioc_filled = fill_count > 0
                        stop_log.info("FILL POLL  order_id=%s  fill_count=%.2f  filled=%s",
                                      exit_order_id, fill_count, ioc_filled)
                        if not ioc_filled:
                            print(f"[STOP] Attempt {attempt+1}: IoC unfilled @ {price_floor}¢")
                    except Exception as poll_err:
                        if "404" in str(poll_err):
                            ioc_filled = False
                        else:
                            ioc_filled = True

                if ioc_filled or attempt >= Config.STOP_EXIT_MAX_RETRIES:
                    if not ioc_filled:
                        pos_snapshot = dict(pos)
                        self._rescue_in_progress = True
                        asyncio.create_task(self._stop_rescue_loop(
                            kalshi, pos_snapshot, data["ticker"], data))
                        self.log("STOP_EXPIRY_RISK", attempt_ctx,
                                 f"Stop exhausted {Config.STOP_EXIT_MAX_RETRIES+1} attempts — rescue loop started")
                    if ioc_filled:
                        pos_snapshot = dict(pos)
                        asyncio.create_task(self._confirm_stop_exit(
                            kalshi, exit_order_id, pos_snapshot, data["ticker"], price_floor))
                    break

            except Exception as e:
                stop_log.error("STOP ATTEMPT %d EXCEPTION  error=%s", attempt + 1, e)
                print(f"[STOP] Attempt {attempt+1} failed: {e}")
                if attempt == Config.STOP_EXIT_MAX_RETRIES:
                    self.log("STOP_FAILED", attempt_ctx,
                             f"All {Config.STOP_EXIT_MAX_RETRIES+1} stop attempts failed: {e}")

        if stop_submitted:
            self._stop_exit_submitted = False
            self._stop_failed_ticker  = data.get("ticker", "")
            if self._rescue_in_progress:
                # Rescue loop owns active_position — keep _stop_exit_in_progress=True
                # so the stop engine doesn't re-fire on every subsequent tick.
                pass
            else:
                self.active_position        = None
                self.session_fills          = 0
                self._stop_exit_in_progress = False
        else:
            print(f"\033[91m[STOP] All attempts raised before API call — retaining position\033[0m")
            self.log("STOP_RETRY", stop_ctx, "Stop submission failed — retaining position")
            self._stop_exit_in_progress = False

    async def _stop_rescue_loop(self, kalshi, position: dict, ticker: str, data: dict):
        side  = position["side"]
        qty   = position["qty"]
        price = Config.STOP_FLOOR_CENTS
        attempt = 0
        while True:
            await asyncio.sleep(Config.STOP_RESCUE_INTERVAL_SEC)
            if not self._rescue_in_progress:
                # Cleared by session roll — market settled, nothing to do.
                return
            try:
                coid = make_client_order_id(ticker, position.get("ml_birth_ts", 0),
                                            "rescue", attempt)
                attempt += 1
                order_kwargs = {
                    "ticker": ticker, "action": "sell", "side": side, "count": qty,
                    "time_in_force": "immediate_or_cancel", "client_order_id": coid,
                }
                if side == "yes":
                    order_kwargs["yes_price"] = price
                else:
                    order_kwargs["no_price"] = price
                resp    = await kalshi.create_order(**order_kwargs)
                oid     = resp.get("order", {}).get("order_id", "")
                await asyncio.sleep(1.0)
                filled  = False
                if oid:
                    order_resp = await kalshi.get_order(oid)
                    fill_count = float(order_resp.get("order", {}).get("fill_count_fp", "0"))
                    filled     = fill_count > 0
                if filled:
                    gross_cost  = position.get("entry_cost",
                                               position["entry_price"] * qty / 100.0)
                    net_fees    = position.get("entry_fees", 0.0)
                    exit_proceeds = qty * price / 100.0
                    pnl = exit_proceeds - gross_cost - net_fees
                    self.risk.record_pnl(pnl)
                    self.log("STOP_CONFIRMED", {
                        "ticker": ticker, "side": side,
                        "entry_price": position["entry_price"], "qty": qty,
                        "order_id": oid,
                        "gross_proceeds": exit_proceeds, "gross_cost": gross_cost,
                        "net_fees": net_fees, "pnl_this_trade": pnl,
                    }, f"Rescue filled: {side.upper()} exit @ {price}¢ | pnl=${pnl:.4f}")
                    self.active_position        = None
                    self.session_fills          = 0
                    self._stop_failed_ticker    = ticker
                    self._rescue_in_progress    = False
                    self._stop_exit_in_progress = False
                    stop_log.info("RESCUE LOOP FILLED  order_id=%s  pnl=$%.4f", oid, pnl)
                    asyncio.create_task(self.risk.sync_live_balance(kalshi, bot_ref=self))
                    return
            except Exception as e:
                stop_log.error("RESCUE LOOP ERROR  %s", e)

    async def _confirm_stop_exit(self, kalshi, order_id: str, position: dict,
                                  ticker: str, submitted_price: int):
        await asyncio.sleep(Config.STOP_CONFIRM_DELAY_SEC)
        try:
            fills_resp = await kalshi.get_fills(order_id=order_id)
            fills      = fills_resp.get("fills", [])
            side       = position["side"]
            qty        = position["qty"]

            # Use helper only for fill_qty and sell_fees.  Kalshi returns side="no" for
            # YES sells (normalises sell-YES → buy-NO internally), so _parse_fill_cost_and_fees
            # would invert yes_price_dollars and produce the wrong exit price.  We compute
            # proceeds here using the position side instead of the fill-level side field.
            _, sell_fees, fill_qty, _ = _parse_fill_cost_and_fees(fills, "sell")

            sell_fills = [f for f in fills if f.get("action") == "sell"]
            if side == "yes":
                sell_proceeds = sum(
                    float(f.get("count_fp", 0)) * float(f.get("yes_price_dollars") or 0)
                    for f in sell_fills
                )
            else:
                sell_proceeds = sum(
                    float(f.get("count_fp", 0)) *
                    float(f.get("no_price_dollars") or
                          (1.0 - float(f.get("yes_price_dollars") or 0)))
                    for f in sell_fills
                )

            avg_exit_price = round((sell_proceeds / fill_qty) * 100) if fill_qty > 0 else submitted_price

            gross_cost = position.get("entry_cost", position["entry_price"] * qty / 100.0)
            net_fees   = position.get("entry_fees", 0.0) + sell_fees
            pnl        = sell_proceeds - gross_cost - net_fees

            self.risk.record_pnl(pnl)
            self._stop_failed_ticker = ""
            self.log("STOP_CONFIRMED", {
                "ticker": ticker, "side": side, "entry_price": position["entry_price"],
                "qty": qty, "order_id": order_id,
                "gross_proceeds": sell_proceeds, "gross_cost": gross_cost,
                "net_fees": net_fees, "pnl_this_trade": pnl,
            }, f"Stop confirmed: {side.upper()} exit @ {avg_exit_price}¢ | pnl=${pnl:.4f}")
            await self.risk.sync_live_balance(kalshi, bot_ref=self)
            asyncio.create_task(self._reconcile_stop_exit(
                kalshi, order_id, position, ticker, pnl, sell_proceeds, gross_cost, net_fees
            ))
        except Exception as e:
            stop_log.error("CONFIRM STOP EXIT EXCEPTION  order_id=%s  error=%s", order_id, e)
            self.log("ERROR", {"ticker": ticker},
                     f"_confirm_stop_exit failed for {order_id}: {e}")

    async def _reconcile_stop_exit(
        self, kalshi, order_id: str, position: dict, ticker: str,
        estimated_pnl: float, est_proceeds: float, est_cost: float, est_fees: float,
    ):
        await asyncio.sleep(10.0)
        side = position["side"]
        qty  = position["qty"]
        try:
            fills_resp = await kalshi.get_fills(order_id=order_id)
            fills      = fills_resp.get("fills", [])
            _, actual_fees, fill_qty, _ = _parse_fill_cost_and_fees(fills, "sell")

            sell_fills = [f for f in fills if f.get("action") == "sell"]
            if side == "yes":
                actual_proceeds = sum(
                    float(f.get("count_fp", 0)) * float(f.get("yes_price_dollars") or 0)
                    for f in sell_fills
                )
            else:
                actual_proceeds = sum(
                    float(f.get("count_fp", 0)) *
                    float(f.get("no_price_dollars") or
                          (1.0 - float(f.get("yes_price_dollars") or 0)))
                    for f in sell_fills
                )

            if fill_qty == 0:
                return

            actual_pnl = actual_proceeds - est_cost - (position.get("entry_fees", 0.0) + actual_fees)
            delta      = actual_pnl - estimated_pnl
            diverged   = abs(delta) >= Config.RECONCILE_DIVERGENCE_THRESHOLD

            reconcile_ctx = {
                "ticker": ticker, "side": side,
                "entry_price": position["entry_price"], "qty": qty,
                "order_id": order_id,
                "gross_proceeds":          actual_proceeds,
                "gross_cost":              est_cost,
                "net_fees":                position.get("entry_fees", 0.0) + actual_fees,
                "pnl_this_trade":          actual_pnl,
                "reconcile_estimated_pnl": estimated_pnl,
                "reconcile_actual_pnl":    actual_pnl,
                "reconcile_delta":         delta,
                "settlement_source":       "stop_exit",
            }
            flag = "⚠️ DIVERGENCE" if diverged else "OK"
            self.log("RECONCILE", reconcile_ctx,
                     f"{flag} | estimated=${estimated_pnl:+.4f} actual=${actual_pnl:+.4f} "
                     f"delta=${delta:+.4f}")
            if diverged:
                stop_log.warning("RECONCILE DIVERGENCE %s: estimated=%+.4f actual=%+.4f delta=%+.4f",
                                 ticker, estimated_pnl, actual_pnl, delta)
        except Exception as e:
            stop_log.error("RECONCILE STOP EXIT ERROR  order_id=%s  error=%s", order_id, e)

    # ── Order recovery ────────────────────────────────────────────────────────

    async def _recover_missing_order(self, order_id: str, data: dict):
        logger = setup_logging()
        if not order_id or Config.PAPER_MODE:
            return
        kalshi = self._kalshi_ref
        if kalshi is None:
            return
        await asyncio.sleep(2.0)
        try:
            resp       = await kalshi.get_order(order_id)
            order_data = resp.get("order", {})
            status     = order_data.get("status", "")
            fill_count = float(order_data.get("fill_count_fp", "0"))

            if fill_count > 0 and self.active_position is None:
                ao = self.active_order or {}
                if ao.get("order_id") != order_id:
                    ao = {}
                side    = order_data.get("side", ao.get("side", "yes"))
                qty     = int(float(order_data.get("remaining_count_fp",
                                                    order_data.get("count", fill_count))))
                price_raw = order_data.get("yes_price_dollars") if side == "yes" \
                            else order_data.get("no_price_dollars")
                price_c = round(float(price_raw or 0) * 100)

                if self.active_order is not None and self.active_order.get("order_id") != order_id:
                    return

                self.active_position = {
                    "ticker":          order_data.get("ticker", data.get("ticker", "")),
                    "side":            side, "qty": qty, "entry_price": price_c,
                    "order_id":        order_id,
                    "client_order_id": order_data.get("client_order_id", ao.get("client_order_id", "")),
                    "ml_direction":    ao.get("ml_direction"),
                    "ml_confidence":   ao.get("ml_confidence", 0.0),
                    "ml_proba_up":     ao.get("ml_proba_up", 0.0),
                    "ml_birth_ts":     ao.get("birth_ts", 0),
                    "signal_age_min":  ao.get("signal_age_min", 0.0),
                    "entry_cost":      qty * price_c / 100.0,
                    "entry_fees":      0.0,
                }
                if self.active_order is not None and self.active_order.get("order_id") == order_id:
                    self.active_order = None
                self._init_stop(price_c, data.get("minutes_left", 0.0))
                self.log("FILL_CONFIRMED", {
                    "ticker": self.active_position["ticker"], "side": side,
                    "entry_price": price_c, "qty": qty, "order_id": order_id,
                    "order_status": "filled", "kalshi_fill_qty": qty, "kalshi_fill_price": price_c,
                }, f"WS fill RECOVERED via REST: {side.upper()} {qty}x @ {price_c}c")
                print(f"\033[93m[RECOVER] {side.upper()} {qty}x @ {price_c}c\033[0m")
                asyncio.create_task(self._fetch_fill_costs(order_id, self.active_position["ticker"]))
            else:
                if self._last_recovered_order_id == order_id:
                    return
                if self.active_order is not None and self.active_order.get("order_id") != order_id:
                    return
                self._last_recovered_order_id = order_id
                self.active_order             = None
                self.session_fills           -= 1
                self.acted_on_ml_birth_ts    = None
                self._save_birth_time(None)
                self.log("ORDER_UNFILLED", {
                    "ticker": data.get("ticker", ""), "order_id": order_id,
                    "order_status": f"gone_unfilled_status={status}",
                }, f"Order {order_id} gone unfilled (status={status}) — cleared")
                print(f"\033[91m[RECOVER] Order {order_id} gone unfilled — cleared\033[0m")
        except Exception as e:
            if "404" in str(e):
                if self._last_recovered_order_id == order_id:
                    return
                if self.active_order is not None and self.active_order.get("order_id") != order_id:
                    return
                self._last_recovered_order_id = order_id
                self.active_order             = None
                self.session_fills           -= 1
                self.acted_on_ml_birth_ts    = None
                self._save_birth_time(None)
                self.log("ORDER_UNFILLED", {
                    "ticker": data.get("ticker", ""), "order_id": order_id,
                    "order_status": "404_on_recovery_fetch",
                }, f"Order {order_id} 404'd — assumed dead, cleared")
                print(f"\033[91m[RECOVER] Order {order_id} 404'd — cleared\033[0m")
            else:
                print(f"[RECOVER] Could not resolve missing order {order_id}: {e}")

    async def _poll_order_status(self, order_id: str, data: dict):
        await asyncio.sleep(Config.POST_ONLY_CANCEL_POLL_SEC)
        if self.active_order is None or self.active_order.get("order_id") != order_id:
            return
        kalshi = self._kalshi_ref
        if kalshi is None:
            return
        try:
            resp       = await kalshi.get_order(order_id)
            order_data = resp.get("order", {})
            status     = order_data.get("status", "")
            fill_count = float(order_data.get("fill_count_fp", "0"))
            if status == "resting":
                return
            if fill_count > 0:
                return
            print(f"\033[93m[POLL] Post-only order {order_id} canceled (status={status}) — recovering\033[0m")
            asyncio.create_task(self._recover_missing_order(order_id, data))
        except Exception as e:
            if "404" in str(e):
                asyncio.create_task(self._recover_missing_order(order_id, data))

    # ── Active order management (penny-jumping + TTL + taker escalation) ──────

    async def _manage_active_order(self, kalshi, data):
        if self.active_order is None:
            return
        ao        = self.active_order
        side      = ao["side"]
        now       = time.time()
        time_left = data.get("minutes_left", 0.0)

        if not ao.get("order_id"):
            self.log("ORDER_UNFILLED", {
                "ticker": ao.get("ticker", data.get("ticker", "")),
                "order_id": "", "client_order_id": ao.get("client_order_id", ""),
                "order_status": "blank_order_id_never_placed",
            }, "Blank order_id — order never confirmed; clearing")
            self.active_order             = None
            self.session_fills           -= 1
            self.acted_on_ml_birth_ts    = None
            self._save_birth_time(None)
            return

        if data["ticker"] != ao["ticker"]:
            if not Config.PAPER_MODE:
                try:
                    await kalshi.cancel_order(ao["order_id"])
                except Exception:
                    pass
            self.log("ORDER_UNFILLED", {
                **data, "order_id": ao["order_id"],
                "client_order_id": ao["client_order_id"],
                "order_status": "expired_on_roll",
            }, f"Order expired on session roll → {data['ticker']}")
            self.active_order             = None
            self.session_fills           -= 1
            self.acted_on_ml_birth_ts    = None
            self._save_birth_time(None)
            return

        time_since_placed = now - ao.get("posted_at", now)

        # TTL check — cancel stale maker orders (ENTRY_TTL_SECONDS)
        if time_since_placed >= Config.ENTRY_TTL_SECONDS and not Config.PAPER_MODE:
            try:
                await kalshi.cancel_order(ao["order_id"])
                print(f"\033[93m[TTL] Order {ao['order_id']} expired after "
                      f"{time_since_placed:.0f}s — canceled\033[0m")
            except Exception as e:
                print(f"[TTL] Cancel failed: {e}")
            self.log("ORDER_UNFILLED", {
                **data, "side": side, "order_id": ao["order_id"],
                "client_order_id": ao["client_order_id"],
                "order_status": f"ttl_expired_{time_since_placed:.0f}s",
            }, f"Order TTL expired after {time_since_placed:.0f}s — canceled")
            self.active_order             = None
            self.session_fills           -= 1
            self.acted_on_ml_birth_ts    = None
            self._save_birth_time(None)
            return

        # Taker escalation when time is short
        if Config.TAKER_FORCE_MIN_LEFT > 0 and time_left < Config.TAKER_FORCE_MIN_LEFT \
                and time_since_placed >= Config.AMEND_COOLDOWN_SEC:
            raw_ask = data["ask_yes"] if side == "yes" else data["ask_no"]
            ask     = min(raw_ask, Config.MAKER_MAX_ENTRY_PRICE)
            if ask < 10:
                self.log("ORDER_UNFILLED", {
                    **data, "side": side, "entry_price": raw_ask,
                    "order_id": ao.get("order_id", ""),
                    "order_status": "taker_escalation_skipped_no_book",
                }, f"Taker escalation skipped: ask {ask}¢ too low")
                return

            coid = make_client_order_id(ao["ticker"], ao["birth_ts"], "entry_taker")
            if Config.PAPER_MODE:
                entry_cost = ao["qty"] * (ask / 100.0)
                self.risk.paper_balance -= entry_cost
                self.active_position = {
                    "ticker": ao["ticker"], "side": side, "qty": ao["qty"],
                    "entry_price": ask, "order_id": "", "client_order_id": coid,
                    "ml_direction": ao.get("ml_direction"), "ml_confidence": ao.get("ml_confidence", 0.0),
                    "ml_proba_up": ao.get("ml_proba_up", 0.0),
                    "ml_birth_ts": ao["birth_ts"], "signal_age_min": ao.get("signal_age_min", 0.0),
                    "entry_cost": entry_cost, "entry_fees": 0.0,
                }
                self.active_order = None
                self._init_stop(ask, time_left)
                self.log("ORDER_ESCALATED", {
                    **data, "side": side, "entry_price": ask, "qty": ao["qty"],
                    "client_order_id": coid,
                }, f"PAPER taker fill: {side.upper()} @ {ask}¢ x{ao['qty']}")
            else:
                try:
                    await kalshi.cancel_order(ao["order_id"])
                    order_kwargs = {
                        "ticker": ao["ticker"], "action": "buy", "side": side,
                        "count": ao["qty"], "time_in_force": "immediate_or_cancel",
                        "client_order_id": coid,
                    }
                    if side == "yes":
                        order_kwargs["yes_price"] = ask
                    else:
                        order_kwargs["no_price"] = ask
                    resp    = await kalshi.create_order(**order_kwargs)
                    new_oid = resp.get("order", {}).get("order_id", "")
                    if self.active_order is None:
                        return
                    self.active_order["order_id"]        = new_oid
                    self.active_order["client_order_id"] = coid
                    self.active_order["posted_price"]    = ask
                    self.log("ORDER_ESCALATED", {
                        **data, "side": side, "entry_price": ask, "qty": ao["qty"],
                        "order_id": new_oid, "client_order_id": coid,
                    }, f"Taker IoC: {side.upper()} @ {ask}¢ x{ao['qty']} | {time_left:.1f}m left")
                except Exception as e:
                    esc_log.error("ESCALATION EXCEPTION  error=%s  maker_order_id=%s", e, ao.get("order_id", ""))
                    self.log("ERROR", data, f"Taker escalation failed: {e}")
                    if "404" in str(e):
                        asyncio.create_task(self._recover_missing_order(ao["order_id"], data))
            return

        # Penny-jump repricing
        current_bid      = data["raw_yes_bid"] if side == "yes" else data["raw_no_bid"]
        drift            = abs(current_bid - ao["posted_price"])
        time_since_amend = now - ao.get("last_amend_ts", now)

        if drift >= Config.REPRICE_THRESHOLD and time_since_amend >= Config.AMEND_COOLDOWN_SEC:
            best_ask  = data["ask_yes"] if side == "yes" else data["ask_no"]
            new_price = max(1, min(Config.MAKER_MAX_ENTRY_PRICE,
                                   (current_bid + 1) if best_ask > (current_bid + 1) else current_bid))
            if new_price == ao["posted_price"]:
                return
            if Config.PAPER_MODE:
                old_price = ao["posted_price"]
                ao["posted_price"]   = new_price
                ao["last_amend_ts"]  = now
                self.log("ORDER_AMENDED", {
                    **data, "side": side, "entry_price": new_price, "qty": ao["qty"],
                    "order_id": ao["order_id"], "client_order_id": ao["client_order_id"],
                }, f"PAPER repriced {old_price}¢ → {new_price}¢")
                return
            try:
                old_price  = ao["posted_price"]
                amend_kwargs: dict = {
                    "order_id": ao["order_id"],
                    "ticker":   ao["ticker"],
                    "side":     side,
                    "action":   "buy",
                    "count":    ao["qty"],
                }
                if side == "yes":
                    amend_kwargs["yes_price"] = new_price
                else:
                    amend_kwargs["no_price"] = new_price
                await kalshi.amend_order(**amend_kwargs)
                ao["posted_price"]  = new_price
                ao["last_amend_ts"] = now
                self.log("ORDER_AMENDED", {
                    **data, "side": side, "entry_price": new_price, "qty": ao["qty"],
                    "order_id": ao["order_id"], "client_order_id": ao["client_order_id"],
                }, f"Repriced {old_price}¢ → {new_price}¢ (drift={drift}¢)")
            except Exception as e:
                esc_log.warning("AMEND FAILED  order_id=%s  error=%s", ao["order_id"], e)
                if "404" in str(e):
                    asyncio.create_task(self._recover_missing_order(ao["order_id"], data))

    def _build_ladder(self, side: str, qty: int, depth: list, birth_ts: float, ticker: str) -> list:
        orders    = []
        remaining = qty
        for i, (price_c, avail_qty) in enumerate(depth):
            if not (1 <= price_c <= Config.MAKER_MAX_ENTRY_PRICE):
                continue
            slice_qty = min(remaining, max(1, int(avail_qty * Config.LADDER_FILL_FRACTION)))
            coid      = make_client_order_id(ticker, birth_ts, "entry", i)
            order     = {
                "ticker": ticker, "action": "buy", "side": side,
                "count": slice_qty, "post_only": True, "client_order_id": coid,
            }
            if side == "yes":
                order["yes_price"] = price_c
            else:
                order["no_price"] = price_c
            orders.append(order)
            remaining -= slice_qty
            if remaining <= 0:
                break
        return orders

    # ── ML inference gate ─────────────────────────────────────────────────────

    async def _run_ml_inference(self, data: dict) -> tuple[bool, str]:
        """
        Runs the ML inference pipeline exactly ONCE per 15-minute session,
        within ML_INFERENCE_WINDOW_MIN minutes of session open.

        Features are read from data/btc_1min.csv (written by the background collector)
        and processed by BTCMinuteProcessor — the same pipeline used at training time.

        Gates (in order):
          1. Session-open window  — only fire within first N minutes of session
          2. Kalshi spread gate   — need a liquid Kalshi market to trade into
          3. ATR volatility gate  — reject if live ATR_14 outside [ATR_MIN, ATR_MAX]
          4. Feature build + predict_proba via BTCMinuteProcessor
          5. Confidence tau gate  — reject low-conviction predictions
          6. Moneyness gate       — reject if Kalshi strike is too far against direction

        Returns (should_trade: bool, filter_reason: str).
        The ml_direction/ml_confidence stored in SharedState are the definitive
        session signal regardless of whether should_trade is True — heartbeat
        and log always show them.
        """
        minutes_left = data["minutes_left"]
        session_open_threshold = Config.TIME_ENTRY_MIN_MIN - Config.ML_INFERENCE_WINDOW_MIN
        # Snapshot tau once for this entire call — used for gate check AND logging so
        # both always reflect the same value, even if config.json is hot-reloaded mid-call.
        tau_at_inference = Config.ML_CONFIDENCE_TAU

        if self.shared.ml_session_fired:
            # Inference already ran this session — reuse the cached prediction.
            # Skip the CSV read/feature engineering entirely to avoid per-tick CPU spikes.
            direction  = self.shared.ml_direction
            confidence = self.shared.ml_confidence
            proba_up   = self.shared.ml_proba_up
        else:
            # Gate 1: Inference must happen within the opening window of the session.
            if minutes_left < session_open_threshold:
                return False, f"inference_window_expired_{minutes_left:.1f}m_left"

            # Gate 2: Kalshi spread liquidity check
            kalshi_spread = data["ask_yes"] - data["raw_yes_bid"]
            if kalshi_spread > Config.ML_SPREAD_MAX_CENTS:
                return False, f"kalshi_spread_too_wide_{kalshi_spread}c"

            # Allow the background collector to finish flushing the current minute bar.
            await asyncio.sleep(2.0)

            # Gates 3–4: Read CSV, compute ATR, engineer features, run model
            try:
                from ml.preprocess import BTCMinuteProcessor, PreprocessingConfig

                csv_path = project_root / "data" / "btc_1min.csv"
                raw_df = (
                    pd.read_csv(csv_path, parse_dates=["timestamp"])
                    .tail(100)
                    .reset_index(drop=True)
                )
                raw_df["symbol"] = "BTC_USDT"

                # Gate 3: ATR volatility gate (uses raw OHLC — must run before normalization)
                current_atr = _calculate_live_atr_14(raw_df)
                if current_atr < Config.ATR_MIN or current_atr > Config.ATR_MAX:
                    return False, f"atr_outside_range_{current_atr:.2f}"

                # Normalize order book levels to price-relative distances / imbalances.
                # Must match BTCMinuteProcessor.engineer_features() in ml/preprocess.py
                # exactly so live features are identical to training features.
                raw_df = _normalize_ob_levels(raw_df)

                # Gate 4: Feature engineering via BTCMinuteProcessor
                _dummy = Path("/dev/null")
                proc_config = PreprocessingConfig(
                    macro_csv_path=_dummy,
                    micro_data_dir=_dummy,
                    output_dir=_dummy,
                    save_debug_info=False,
                )
                processor = BTCMinuteProcessor(csv_path=csv_path, config=proc_config)
                processor.data = raw_df  # bypass load_data() / clean_data()
                feat_df_full = processor.engineer_features()

                # If the schema contains features not in the current CSV (e.g. Bybit
                # columns during a transition period), fill with NaN so the imputer
                # can substitute training means rather than crashing on KeyError.
                missing = [f for f in ML.features if f not in feat_df_full.columns]
                if missing:
                    log.warning("[ML] %d schema features missing from CSV — imputer filling", len(missing))
                    for mf in missing:
                        feat_df_full[mf] = float("nan")

                feat_row = feat_df_full.iloc[[-1]][ML.features]

                direction, confidence, proba_up = ML.predict(feat_row)
                self.shared.update_ml(direction, confidence, proba_up)

                dir_str = "UP" if direction == 1 else "DN"
                self.log("ML_INFERENCE", {
                    "ticker":        data["ticker"],
                    "ml_direction":  direction,
                    "ml_confidence": confidence,
                    "ml_proba_up":   proba_up,
                    "ml_tau":        tau_at_inference,
                    "ml_birth_ts":   self.shared.ml_birth_ts,
                    "atr_14":        current_atr,
                }, f"{dir_str} | conf={confidence:.4f} | proba_up={proba_up:.4f} | "
                   f"tau={tau_at_inference:.3f} | window={minutes_left:.1f}m left | "
                   f"atr={current_atr:.2f} | BTC={self.shared.latest_btc:.2f}")

            except Exception as e:
                setup_logging().error(f"ML inference error: {e}")
                return False, f"inference_exception_{e}"

        # Gate 5: Confidence threshold — use tau_at_inference (snapshotted at function entry)
        # so the gate value always matches what was logged in the ML_INFERENCE row.
        if confidence < tau_at_inference:
            return False, f"confidence_below_tau_{confidence:.4f}"

        # Gate 6 (always checked, even for cached signals): Moneyness gate.
        # The model predicts BTC price direction, but the Kalshi strike anchors
        # the contract's moneyness. If the strike is already deeply against our
        # direction, the edge is lost even if the model is right about direction.
        #
        # For YES (BTC goes UP): we need Strike <= Spot + threshold
        #   disadvantage_bps = (Strike - Spot) / Spot * 10000
        #   Positive = strike above spot (bad for YES, good for NO)
        #
        # For NO  (BTC goes DOWN): we need Strike >= Spot - threshold
        #   disadvantage_bps = (Spot - Strike) / Spot * 10000
        #   Positive = strike below spot (bad for NO, good for YES)
        spot   = self.shared.latest_btc
        strike = data["strike"]
        if spot > 0 and strike > 0:
            if direction == 1:  # YES — we need BTC to be above strike at expiry
                disadvantage_bps = (strike - spot) / spot * 10_000.0
            else:               # NO  — we need BTC to be below strike at expiry
                disadvantage_bps = (spot - strike) / spot * 10_000.0

            if disadvantage_bps > Config.MAX_STRIKE_DISADVANTAGE_BPS:
                dir_str = "YES" if direction == 1 else "NO"
                return False, (f"moneyness_gate_{dir_str}_strike={strike:.0f}"
                               f"_spot={spot:.0f}_disadv={disadvantage_bps:.1f}bps")

        return True, ""

    # ── Main tick handler ─────────────────────────────────────────────────────

    async def on_tick(self, kalshi, data, ob_deep_stale: bool = False):
        """
        Called every ~1 second from tick_loop.
        Gate order:
          1. Circuit breaker
          2. Active order management (penny-jump, TTL, taker)
          3. Stop check on active position
          4. Entry filters (session, timing, OB staleness)
          5. ML inference (only when filters pass)
          6. Place maker order
        """
        btc = self.shared.latest_btc

        ob_stale = ob_deep_stale or (time.time() - self.ob.last_update_ts) > Config.MAX_ORDERBOOK_STALE_SEC

        # ── Circuit breaker ──────────────────────────────────────────────────
        if not self.circuit_breaker_open:
            if abs(self.risk.rolling_24h_loss()) >= Config.MAX_DAILY_LOSS:
                self.circuit_breaker_open = True
        if self.circuit_breaker_open:
            if not self._circuit_breaker_logged:
                self._circuit_breaker_logged = True
                self.log("CIRCUIT_BREAKER", {
                    "ticker": data["ticker"], "time_left": data["minutes_left"],
                    "btc": btc, "rolling_24h_loss": self.risk.rolling_24h_loss(),
                }, f"CIRCUIT BREAKER OPEN — 24h loss "
                   f"${abs(self.risk.rolling_24h_loss()):.2f} >= limit ${Config.MAX_DAILY_LOSS:.2f}")
                print(f"\033[91m[CIRCUIT BREAKER] Daily loss limit hit — entries halted.\033[0m")
            if self.active_order is not None:
                await self._manage_active_order(kalshi, data)
            if self.active_position is not None:
                if self.active_position["ticker"] == data["ticker"]:
                    pos_side    = self.active_position["side"]
                    current_bid = data["raw_yes_bid"] if pos_side == "yes" else data["raw_no_bid"]
                    if current_bid > 0 and self._check_stop(current_bid):
                        if not self._stop_exit_in_progress:
                            self._stop_exit_in_progress = True
                            await self._stop_exit(kalshi, data)
            return

        # ── ML state for heartbeat / logging ─────────────────────────────────
        ml_direction  = self.shared.ml_direction
        ml_confidence = self.shared.ml_confidence
        ml_proba_up   = self.shared.ml_proba_up
        ml_birth_ts   = self.shared.ml_birth_ts
        signal_age_min = (time.time() - ml_birth_ts) / 60.0 if ml_birth_ts > 0 else 999.0

        ctx = {
            "ticker":        data["ticker"], "time_left": data["minutes_left"],
            "btc":           btc, "strike": data["strike"],
            "raw_yes_bid":   data["raw_yes_bid"], "raw_no_bid": data["raw_no_bid"],
            "ask_yes":       data["ask_yes"], "ask_no": data["ask_no"],
            "yes_liq":       data["yes_liq"], "no_liq": data["no_liq"], "obi": data["obi"],
            "ml_direction":  ml_direction, "ml_confidence": ml_confidence,
            "ml_proba_up":   ml_proba_up,  "ml_birth_ts": ml_birth_ts,
            "signal_age_min": signal_age_min,
        }

        # ── Active order management ───────────────────────────────────────────
        if self.active_order is not None:
            await self._manage_active_order(kalshi, data)

        # ── Stop check on active position ─────────────────────────────────────
        if self.active_position is not None:
            if self.active_position["ticker"] == data["ticker"]:
                pos_side    = self.active_position["side"]
                current_bid = data["raw_yes_bid"] if pos_side == "yes" else data["raw_no_bid"]
                if current_bid > 0 and self._check_stop(current_bid):
                    if not self._stop_exit_in_progress:
                        self._stop_exit_in_progress = True
                        await self._stop_exit(kalshi, data)


        # ── Heartbeat ────────────────────────────────────────────────────────
        if time.time() - self.last_heartbeat_ts > Config.HEARTBEAT_INTERVAL_SEC:
            bank = self.risk.paper_balance if Config.PAPER_MODE else self.risk.real_balance
            ctx["bankroll"]     = bank
            ctx["ob_stale"]     = int(ob_stale)
            ctx["has_position"] = self.active_position is not None
            ctx["has_order"]    = self.active_order is not None
            self.log("HRTBT", ctx)
            self.last_heartbeat_ts = time.time()

        # ── Pre-inference filters ─────────────────────────────────────────────
        if self.session_fills >= Config.MAX_FILLS_PER_SESSION:
            return
        if self.active_position or self.active_order:
            return
        if self._stop_failed_ticker and self._stop_failed_ticker == data["ticker"]:
            ctx["filter_reason"] = "stop_failed_this_session_no_reentry"
            return
        if data["minutes_left"] < Config.TIME_ENTRY_MAX_MIN:
            ctx["filter_reason"] = f"too_close_to_expiry_{data['minutes_left']:.1f}m"
            return
        if ob_stale:
            ctx["filter_reason"] = "orderbook_stale"
            return
        # Session-fill guard: if we already traded on this session's ML signal, stop.
        # (The ML signal itself is now locked per-session inside _run_ml_inference;
        #  this guard prevents placing a second order if the first one was cancelled
        #  and another tick somehow slips through before session_fills resets.)
        if self.acted_on_ml_birth_ts is not None and \
                self.acted_on_ml_birth_ts == self.shared.ml_birth_ts:
            ctx["filter_reason"] = "already_acted_on_this_session_signal"
            return

        # ── ML INFERENCE ──────────────────────────────────────────────────────
        should_trade, filter_reason = await self._run_ml_inference(data)
        if not should_trade:
            ctx["filter_reason"] = filter_reason
            return

        # Re-read shared state after inference (just updated)
        ml_direction  = self.shared.ml_direction
        ml_confidence = self.shared.ml_confidence
        ml_proba_up   = self.shared.ml_proba_up
        ml_birth_ts   = self.shared.ml_birth_ts
        signal_age_min = (time.time() - ml_birth_ts) / 60.0

        side = "yes" if ml_direction == 1 else "no"

        best_bid    = data["raw_yes_bid"] if side == "yes" else data["raw_no_bid"]
        best_ask    = data["ask_yes"]     if side == "yes" else data["ask_no"]
        maker_price = max(1, min(Config.MAKER_MAX_ENTRY_PRICE,
                                 (best_bid + 1) if best_ask > (best_bid + 1) else best_bid))

        ctx["side"]           = side
        ctx["entry_price"]    = maker_price
        ctx["ml_direction"]   = ml_direction
        ctx["ml_confidence"]  = ml_confidence
        ctx["ml_proba_up"]    = ml_proba_up
        ctx["ml_birth_ts"]    = ml_birth_ts
        ctx["signal_age_min"] = signal_age_min

        if maker_price < 1 or maker_price > Config.MAKER_MAX_ENTRY_PRICE:
            ctx["filter_reason"] = f"price_out_of_range_{maker_price}c"
            return

        qty = self.risk.calculate_qty(maker_price)
        if qty < 1:
            ctx["filter_reason"] = "qty_zero_insufficient_bankroll"
            return

        ctx["qty"] = qty
        self.acted_on_ml_birth_ts = ml_birth_ts
        self._save_birth_time(ml_birth_ts)
        self.session_fills += 1

        # ── PAPER MODE fast path ───────────────────────────────────────────────
        if Config.PAPER_MODE:
            entry_cost = qty * (maker_price / 100.0)
            self.risk.paper_balance -= entry_cost
            self.active_position = {
                "ticker": data["ticker"], "side": side, "qty": qty,
                "entry_price": maker_price, "order_id": "", "client_order_id": "",
                "ml_direction": ml_direction, "ml_confidence": ml_confidence,
                "ml_proba_up": ml_proba_up,
                "ml_birth_ts": ml_birth_ts, "signal_age_min": round(signal_age_min, 2),
                "entry_cost": entry_cost, "entry_fees": 0.0,
            }
            self._init_stop(maker_price, data["minutes_left"])
            self.log("PAPER_BUY", ctx,
                     f"FILL: {side.upper()} @ {maker_price}¢ x{qty} | "
                     f"conf={ml_confidence:.4f} | Stop: trail={self._stop_trail}¢")
            return

        # ── LIVE order placement ───────────────────────────────────────────────
        try:
            depth = data.get("yes_depth" if side == "yes" else "no_depth", [])

            if qty >= Config.LADDER_THRESHOLD and depth:
                orders = self._build_ladder(side, qty, depth, ml_birth_ts, data["ticker"])
                if not orders:
                    raise ValueError("Ladder produced no valid slices")
                resp     = await kalshi.batch_create_orders(orders)
                first    = resp.get("orders", [{}])[0]
                order_id = first.get("order", {}).get("order_id", "")
                coid     = orders[0]["client_order_id"]
                total_placed = sum(o["count"] for o in orders)
                if total_placed < qty:
                    qty = total_placed
                _now = time.time()
                self.active_order = {
                    "order_id": order_id, "client_order_id": coid,
                    "posted_price": maker_price, "posted_at": _now, "last_amend_ts": _now,
                    "ticker": data["ticker"], "side": side, "qty": qty, "birth_ts": ml_birth_ts,
                    "ml_direction": ml_direction, "ml_confidence": ml_confidence,
                    "ml_proba_up": ml_proba_up, "signal_age_min": round(signal_age_min, 2),
                }
                self._init_stop(maker_price, data["minutes_left"])
                self.log("LIVE_BUY", {**ctx, "order_id": order_id, "client_order_id": coid},
                         f"LADDER: {side.upper()} @ multi-level x{qty} | {len(orders)} slices | "
                         f"conf={ml_confidence:.4f}")
                if order_id:
                    asyncio.create_task(self._poll_order_status(order_id, data))
            else:
                use_taker  = (data["minutes_left"] < Config.TAKER_ENTRY_MIN_LEFT
                              and Config.TAKER_ENTRY_MIN_LEFT > 0)
                entry_price = (min(data["ask_yes"] if side == "yes" else data["ask_no"],
                                   Config.MAKER_MAX_ENTRY_PRICE)
                               if use_taker else maker_price)
                if use_taker and entry_price < 10:
                    ctx["filter_reason"] = f"taker_ask_too_low_{entry_price}c"
                    self.session_fills     -= 1
                    self.acted_on_ml_birth_ts = None
                    self._save_birth_time(None)
                    return

                coid = make_client_order_id(data["ticker"], ml_birth_ts, "entry")
                order_kwargs: dict = {
                    "ticker": data["ticker"], "action": "buy", "side": side,
                    "count": qty, "client_order_id": coid,
                }
                if use_taker:
                    order_kwargs["time_in_force"] = "immediate_or_cancel"
                else:
                    order_kwargs["post_only"] = True
                if side == "yes":
                    order_kwargs["yes_price"] = entry_price
                else:
                    order_kwargs["no_price"] = entry_price

                order_id = ""
                for _attempt in range(3):
                    resp     = await kalshi.create_order(**order_kwargs)
                    order_id = resp.get("order", {}).get("order_id", "")
                    if order_id:
                        break
                    await asyncio.sleep(0.05)
                    coid = make_client_order_id(data["ticker"], ml_birth_ts, "entry", _attempt + 1)
                    order_kwargs["client_order_id"] = coid

                _now = time.time()
                self.active_order = {
                    "order_id": order_id, "client_order_id": coid,
                    "posted_price": entry_price, "posted_at": _now, "last_amend_ts": _now,
                    "ticker": data["ticker"], "side": side, "qty": qty, "birth_ts": ml_birth_ts,
                    "ml_direction": ml_direction, "ml_confidence": ml_confidence,
                    "ml_proba_up": ml_proba_up, "signal_age_min": round(signal_age_min, 2),
                }
                self._init_stop(entry_price, data["minutes_left"])

                label = "TAKER IoC" if use_taker else "MAKER"
                self.log("ORDER_RESTING", {
                    **ctx, "order_id": order_id, "client_order_id": coid,
                    "order_status": "ioc_submitted" if use_taker else "resting",
                    "entry_price": entry_price,
                }, f"{label}: {side.upper()} @ {entry_price}¢ x{qty} | "
                   f"conf={ml_confidence:.4f} | trail={self._stop_trail}¢")
                if order_id:
                    asyncio.create_task(self._poll_order_status(order_id, data))

        except Exception as e:
            self.log("ERROR", ctx, f"Order placement failed: {e}")
            self.session_fills        -= 1
            self.acted_on_ml_birth_ts  = None
            self._save_birth_time(None)
            self.active_order          = None

    def _print_no_data_heartbeat(self, live_ob) -> None:
        """Fallback heartbeat when the orderbook snapshot is empty (e.g. during session roll)."""
        bank = self.risk.paper_balance if Config.PAPER_MODE else self.risk.real_balance
        ticker = live_ob.ticker or "N/A"
        nows = "  [NOT CONNECTED]" if not live_ob.ws_connected else ""
        try:
            print(
                f"\033[90m[{'HRTBT':^10}] "
                f"{ticker} | BTC:{self.shared.latest_btc:.2f} | "
                f"ML:--@0.000 | "
                f"Tau:{Config.ML_CONFIDENCE_TAU:.3f} | ATR:[{Config.ATR_MIN}-{Config.ATR_MAX}] | "
                f"Y:--c N:--c OBI:+0.000 | "
                f"Trail:0¢ Best:0¢ Active:False | "
                f"Bank:${bank:.2f} ΔAge:--s{nows}\033[0m"
            )
        except OSError:
            pass
        self.last_heartbeat_ts = time.time()


# ═════════════════════════════════════════════════════════════════════════════
# ASYNC LOOPS
# ═════════════════════════════════════════════════════════════════════════════

async def watch_exchange_loop(ex, bot: StrategyController):
    """
    Streams the BTC/USD ticker from Coinbase (via CCXT) to maintain a live spot price.
    Only the mid-price is needed — for the moneyness gate and heartbeat logs.
    """
    name      = getattr(ex, "id", str(ex))
    connected = False
    while True:
        try:
            if not connected:
                print(f"[WS {name}] Connecting to BTC/USD ticker stream...")
            ticker = await ex.watch_ticker(Config.SYMBOL)
            if not connected:
                print(f"[WS {name}] Ticker stream active.")
                connected = True
            bid = ticker.get("bid")
            ask = ticker.get("ask")
            if bid and ask:
                bot.shared.latest_btc = (bid + ask) / 2.0
        except Exception as e:
            connected = False
            print(f"[WS {name}] Reconnecting: {type(e).__name__}: {e}")
            await asyncio.sleep(5)


async def kalshi_ws_loop(kalshi, bot: StrategyController, live_ob: LiveOrderbook):
    logger = setup_logging()
    bot._kalshi_ref = kalshi

    while True:
        try:
            headers    = await kalshi.get_ws_auth_headers()
            ws_version = tuple(int(x) for x in websockets.__version__.split(".")[:2])
            ws_kwargs  = ({"additional_headers": headers} if ws_version >= (11, 0)
                          else {"extra_headers": headers})
            async with websockets.connect(Config.KALSHI_WS_URL, **ws_kwargs) as ws:
                print("[KWS] Connected to Kalshi WebSocket")
                live_ob.ws_connected = True

                await ws.send(json.dumps({
                    "id": 1, "cmd": "subscribe",
                    "params": {"channels": ["fill"]}
                }))

                msg_id         = 2
                current_ticker = None
                last_ping_ts   = time.time()

                if live_ob.ticker:
                    current_ticker = live_ob.ticker
                    live_ob.ticker_changed.clear()
                    await ws.send(json.dumps({
                        "id": msg_id, "cmd": "subscribe",
                        "params": {"channels": ["orderbook_delta"],
                                   "market_tickers": [current_ticker]}
                    }))
                    msg_id += 1
                    print(f"[KWS] Subscribed to orderbook_delta for {current_ticker}")

                while True:
                    if time.time() - last_ping_ts >= 30.0:
                        try:
                            await ws.ping()
                            last_ping_ts = time.time()
                        except Exception as ping_err:
                            logger.warning(f"[KWS] Ping failed: {ping_err}")
                            break

                    if live_ob.ticker_changed.is_set() and live_ob.ticker != current_ticker:
                        live_ob.ticker_changed.clear()
                        current_ticker = live_ob.ticker
                        if current_ticker:
                            await ws.send(json.dumps({
                                "id": msg_id, "cmd": "subscribe",
                                "params": {"channels": ["orderbook_delta"],
                                           "market_tickers": [current_ticker]}
                            }))
                            msg_id += 1
                            print(f"[KWS] Resubscribed for {current_ticker}")

                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")

                    if msg_type == "orderbook_snapshot":
                        yes_fp = msg.get("msg", {}).get("yes_dollars", [])
                        no_fp  = msg.get("msg", {}).get("no_dollars",  [])
                        await live_ob.apply_snapshot(yes_fp, no_fp)

                    elif msg_type == "orderbook_delta":
                        m     = msg.get("msg", {})
                        side  = "yes" if m.get("side") == "yes" else "no"
                        price = round(float(m.get("price_dollars", m.get("price", 0))) * 100)
                        delta = float(m.get("delta_fp", m.get("delta", 0)))
                        await live_ob.apply_delta(side, price, delta)

                    elif msg_type == "fill":
                        fill_msg = msg.get("msg", {})
                        asyncio.create_task(bot.on_fill(fill_msg))

        except Exception as e:
            live_ob.ws_connected = False
            logger.warning(f"[KWS] Connection dropped: {type(e).__name__}: {e}")
            print(f"[KWS] Reconnecting in 5s...")
            await asyncio.sleep(5)


async def kalshi_market_loop(kalshi, live_ob: LiveOrderbook):
    """Polls Kalshi REST every 15 seconds to detect ticker rollover."""
    logger     = setup_logging()
    prev_close = None
    while True:
        try:
            m_resp = await kalshi.get_markets(series_ticker=Config.SERIES_TICKER, status="open",
                                               max_retries=1)
            now    = datetime.now(timezone.utc)
            future = sorted(
                [m for m in m_resp.get("markets", [])
                 if datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")) > now],
                key=lambda x: x["close_time"]
            )
            if future:
                t          = future[0]
                new_close  = t["close_time"]
                new_ticker = t["ticker"]
                if new_close != prev_close:
                    await live_ob.reset(
                        new_ticker,
                        t.get("floor_strike") or t.get("strike_price") or 0,
                        new_close,
                    )
                    prev_close = new_close
                    print(f"[MKT LOOP] Active ticker: {new_ticker} | close: {new_close}")
        except Exception as e:
            logger.warning(f"[MKT LOOP] Error: {e}")
        await asyncio.sleep(15)


async def tick_loop(kalshi, bot: StrategyController, live_ob: LiveOrderbook):
    """Main ~1-second heartbeat that drives on_tick."""
    logger = setup_logging()
    while True:
        try:
            await asyncio.sleep(1.0)
            update_config_from_file()
            data = live_ob.snapshot()
            ob_deep_stale = (time.time() - live_ob.last_update_ts) > Config.MAX_ORDERBOOK_STALE_SEC * 3
            # Always call on_tick so the heartbeat fires; on_tick gates trading on staleness.
            # Pass ob_deep_stale so on_tick can skip entry/stop logic when OB is too old.
            if data:
                await bot.on_tick(kalshi, data, ob_deep_stale=ob_deep_stale)
            elif ob_deep_stale or time.time() - bot.last_heartbeat_ts > Config.HEARTBEAT_INTERVAL_SEC:
                # No snapshot yet (levels empty after session roll) — still print a heartbeat
                # so the terminal shows the bot is alive during WS reconnect gaps.
                bot._print_no_data_heartbeat(live_ob)
        except Exception as e:
            logger.error(f"[TICK LOOP] {type(e).__name__}: {e}")
            await asyncio.sleep(1)


async def balance_sync_loop(kalshi, bot: StrategyController):
    while True:
        await asyncio.sleep(300)
        await bot.risk.sync_live_balance(kalshi, bot_ref=bot)


async def orderbook_resync_loop(kalshi, live_ob: LiveOrderbook):
    """Periodically resyncs Kalshi OB from REST to prevent drift."""
    logger = setup_logging()
    while True:
        try:
            await asyncio.sleep(30)
            if live_ob.ticker and live_ob.needs_resync():
                ob_resp = await kalshi.get_orderbook(live_ob.ticker, max_retries=1)
                await live_ob.apply_rest_orderbook(ob_resp)
        except Exception as e:
            logger.debug(f"Orderbook resync failed (non-fatal): {e}")


# ═════════════════════════════════════════════════════════════════════════════
# STARTUP RECONCILE  (unchanged from v7)
# ═════════════════════════════════════════════════════════════════════════════

async def startup_reconcile(kalshi, bot: StrategyController):
    if Config.PAPER_MODE:
        return
    logger = setup_logging()
    try:
        print("[STARTUP] Reconciling open orders and positions with Kalshi...")
        orders_resp    = await kalshi.get_orders(status="resting")
        open_orders    = orders_resp.get("orders", [])
        current_ticker = bot.ob.ticker

        restored_order = False
        for o in open_orders:
            print(f"  [ORDER] {o.get('order_id')} | {o.get('ticker')} | "
                  f"{o.get('side')} @ {o.get('yes_price_dollars') or o.get('no_price_dollars')} | "
                  f"status={o.get('status')}")
            if current_ticker and o.get("ticker") == current_ticker and not restored_order:
                side      = o.get("side", "yes")
                price_raw = o.get("yes_price_dollars") if side == "yes" else o.get("no_price_dollars")
                price_c   = round(float(price_raw or 0) * 100)
                qty       = int(float(o.get("remaining_count_fp", o.get("count", 1))))
                bot.active_order = {
                    "order_id":        o["order_id"],
                    "client_order_id": o.get("client_order_id", ""),
                    "posted_price":    price_c,
                    "posted_at":       time.time(),
                    "last_amend_ts":   time.time(),
                    "ticker":          current_ticker,
                    "side":            side,
                    "qty":             qty,
                    "birth_ts":        0.0,
                    "ml_direction":    None,
                    "ml_confidence":   0.0,
                    "ml_proba_up":     0.0,
                    "signal_age_min":  0.0,
                }
                bot.session_fills = 1
                restored_order    = True
                print(f"  [STARTUP] Restored active_order: {side.upper()} @ {price_c}¢ x{qty}")
                logger.info(f"Startup: restored active_order {o['order_id']} {side} @ {price_c}¢ x{qty}")

        positions_resp    = await kalshi.get_positions(count_filter="position")
        positions         = positions_resp.get("market_positions", [])
        restored_position = False
        for p in positions:
            pos_ticker = p.get("ticker", "")
            pos_qty    = float(p.get("position_fp", "0"))
            print(f"  [POSITION] {pos_ticker} | position={pos_qty}")
            if current_ticker and pos_ticker == current_ticker and pos_qty != 0 and not restored_position:
                side = "yes" if pos_qty > 0 else "no"
                qty  = int(abs(pos_qty))
                bot.active_position = {
                    "ticker":          current_ticker,
                    "side":            side,
                    "qty":             qty,
                    "entry_price":     0,
                    "order_id":        "",
                    "client_order_id": "",
                    "ml_direction":    None,
                    "ml_confidence":   0.0,
                    "ml_proba_up":     0.0,
                    "ml_birth_ts":     0,
                    "signal_age_min":  0.0,
                    "entry_cost":      0.0,
                    "entry_fees":      0.0,
                }
                bot.session_fills          = 1
                bot.active_order           = None
                restored_position          = True
                bot._stop_best_bid         = 0
                bot._stop_trail            = Config.STOP_TRAIL_CENTS
                bot._stop_delay_sec        = 0.0
                bot._stop_entry_time       = time.time()
                bot._stop_active           = True
                print(f"  [STARTUP] Restored active_position: {side.upper()} x{qty} on {current_ticker}")
                logger.info(f"Startup: restored active_position {side} x{qty} on {current_ticker}")

        if open_orders and not restored_order:
            print(f"[STARTUP] WARNING: {len(open_orders)} resting order(s) from prior run "
                  f"not matching current ticker — manual review recommended.")
        if positions and not restored_position:
            print(f"[STARTUP] WARNING: {len(positions)} open position(s) from prior run "
                  f"not matching current ticker — manual review recommended.")
    except Exception as e:
        print(f"[STARTUP] Reconciliation failed (non-fatal): {e}")
        logger.warning(f"Startup reconciliation failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

MODEL_RELOAD_INTERVAL_SEC = 300  # poll for new model artifacts every 5 minutes


async def model_watchdog(bot: StrategyController) -> None:
    """
    Polls for artifacts/model_updated.flag every MODEL_RELOAD_INTERVAL_SEC.
    Written by ml/train.py after a successful weekly retrain; signals that
    new model artifacts are ready on disk.

    Hot-reload is deferred while any of these are true:
      - bot has an active position
      - bot has a resting order
      - a stop-exit is in progress
    This ensures the in-flight trade always uses the model it entered with.

    After a successful reload the flag is deleted so subsequent polls are no-ops.
    """
    logger    = setup_logging()
    seen_mtime: float = 0.0

    while True:
        await asyncio.sleep(MODEL_RELOAD_INTERVAL_SEC)
        try:
            if not FLAG_PATH.exists():
                continue

            flag_mtime = FLAG_PATH.stat().st_mtime
            if flag_mtime <= seen_mtime:
                continue  # already processed this version

            # Safety gate — do not reload mid-trade
            if (bot.active_position is not None or
                    bot.active_order is not None or
                    bot._stop_exit_in_progress):
                logger.info("[WATCHDOG] New model flag detected — deferring reload (trade active)")
                continue

            flag_ts = FLAG_PATH.read_text().strip()

            # Load into temporaries first so ML is never in a half-updated state
            new_model   = joblib.load(MODEL_PATH)
            new_scaler  = joblib.load(SCALER_PATH)
            new_imputer = joblib.load(IMPUTER_PATH)
            with open(SCHEMA_PATH) as f:
                new_features = json.load(f)["feature_names"]

            # Assign atomically (single synchronous block — asyncio cannot interrupt)
            ML.model    = new_model
            ML.scaler   = new_scaler
            ML.imputer  = new_imputer
            ML.features = new_features

            # Load optimized confidence threshold written by train.py grid-search
            threshold_path = ARTIFACTS_DIR / "decision_threshold.json"
            new_tau = None
            try:
                if threshold_path.exists():
                    with open(threshold_path) as _tf:
                        _thresh = json.load(_tf)
                    new_tau = float(_thresh["confidence_tau"])
                    Config.ML_CONFIDENCE_TAU = new_tau
            except Exception as _te:
                logger.warning(f"[WATCHDOG] Could not load decision_threshold.json: {_te}")

            # Load win rate from metrics and update Kelly sizing dynamically
            metrics_path = ARTIFACTS_DIR / "metrics_two_class.json"
            new_win_rate = None
            try:
                if metrics_path.exists():
                    with open(metrics_path) as _mf:
                        _metrics = json.load(_mf)
                    new_win_rate = float(_metrics["win_rate"])
                    Config.KELLY_WIN_RATE = new_win_rate
            except Exception as _me:
                logger.warning(f"[WATCHDOG] Could not load metrics_two_class.json: {_me}")

            seen_mtime = flag_mtime
            FLAG_PATH.unlink(missing_ok=True)

            tau_str = f"{new_tau:.4f}" if new_tau is not None else "unchanged"
            wr_str  = f"{new_win_rate:.4f}" if new_win_rate is not None else "unchanged"
            msg = (f"Hot-reloaded model (trained {flag_ts}) | {len(new_features)} features | "
                   f"confidence_tau={tau_str} | kelly_win_rate={wr_str}")
            logger.info(f"[WATCHDOG] {msg}")
            print(f"[WATCHDOG] New confidence_tau={tau_str} | kelly_win_rate={wr_str}")
            bot.log("MODEL_RELOAD", {}, msg)

        except Exception as exc:
            logger.error(f"[WATCHDOG] Reload error: {exc}")


async def main():
    load_config_at_startup()
    rotate_csv_log_if_needed()

    # Load ML artifacts — must happen before event loop logic uses ML
    ML.load()

    kalshi  = KalshiClient()
    shared  = SharedState()
    live_ob = LiveOrderbook()
    bot     = StrategyController(shared, live_ob)
    bot._kalshi_ref = kalshi

    exchanges = {}
    for name in Config.EXCHANGES:
        if not hasattr(ccxt, name):
            print(f"[STARTUP] WARNING: Exchange '{name}' not in ccxt.pro — skipping.")
            continue
        ex = getattr(ccxt, name)({"newUpdates": True})
        exchanges[name] = ex
        print(f"[STARTUP] Exchange loaded: {name}")

    if not exchanges:
        print("[STARTUP] ERROR: No valid exchanges configured.")
        raise SystemExit(1)

    print(f"[STARTUP] Mode: {'PAPER' if Config.PAPER_MODE else '*** LIVE ***'} | v{VERSION}")
    print(f"[STARTUP] ML tau={Config.ML_CONFIDENCE_TAU:.3f} | "
          f"max_entry={Config.MAKER_MAX_ENTRY_PRICE}¢ | "
          f"spread_gate={Config.ML_SPREAD_MAX_CENTS}¢ | "
          f"TTL={Config.ENTRY_TTL_SECONDS}s")

    await bot.risk.sync_live_balance(kalshi, bot_ref=bot)
    if not Config.PAPER_MODE:
        print(f"[STARTUP] Live balance: ${bot.risk.real_balance:.2f}")

    await startup_reconcile(kalshi, bot)

    try:
        m_resp = await kalshi.get_markets(series_ticker=Config.SERIES_TICKER, status="open")
        now    = datetime.now(timezone.utc)
        future = sorted(
            [m for m in m_resp.get("markets", [])
             if datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")) > now],
            key=lambda x: x["close_time"]
        )
        if future:
            t = future[0]
            await live_ob.reset(
                t["ticker"],
                t.get("floor_strike") or t.get("strike_price") or 0,
                t.get("close_time", "")
            )
            live_ob.ticker_changed.clear()
            print(f"[STARTUP] Initial market primed: {t['ticker']}")
    except Exception as e:
        print(f"[STARTUP] WARNING: Could not prime initial ticker: {e}")

    for ex in exchanges.values():
        asyncio.create_task(watch_exchange_loop(ex, bot))

    asyncio.create_task(kalshi_ws_loop(kalshi, bot, live_ob))
    asyncio.create_task(kalshi_market_loop(kalshi, live_ob))
    asyncio.create_task(tick_loop(kalshi, bot, live_ob))
    asyncio.create_task(balance_sync_loop(kalshi, bot))
    asyncio.create_task(orderbook_resync_loop(kalshi, live_ob))
    asyncio.create_task(model_watchdog(bot), name="model-watchdog")

    prev_ticker = live_ob.ticker
    while True:
        try:
            await asyncio.sleep(0.25)
            data = live_ob.snapshot()
            if not data:
                continue

            if prev_ticker and data["ticker"] != prev_ticker:
                rotate_csv_log_if_needed()

                settle_position = bot.active_position
                settle_order_id = (bot.active_order["order_id"] if bot.active_order else
                                   (bot.active_position.get("order_id") if bot.active_position else None))

                asyncio.create_task(bot._bg_settle(
                    kalshi, prev_ticker, shared.latest_btc,
                    bot.prev_strike, settle_position, order_id=settle_order_id,
                ))

                # Session state reset
                bot.active_position         = None
                bot.active_order            = None
                bot.session_fills           = 0
                bot.session_start_time      = time.time()
                bot._stop_best_bid          = 0
                bot._stop_tick_count        = 0
                bot._stop_trail             = 0
                bot._stop_delay_sec         = 0.0
                bot._stop_entry_time        = 0.0
                bot._stop_active            = False
                bot._circuit_breaker_logged = False
                bot._last_recovered_order_id = ""
                bot._stop_exit_submitted    = False
                bot._stop_exit_in_progress  = False
                bot._rescue_in_progress     = False
                bot._stop_failed_ticker     = ""
                bot.acted_on_ml_birth_ts    = None
                bot._save_birth_time(None)
                shared.clear_session_ml()   # Reset ML signal slot for new session

            prev_ticker     = data["ticker"]
            bot.prev_ticker = data["ticker"]
            bot.prev_strike = data["strike"]

        except Exception:
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
