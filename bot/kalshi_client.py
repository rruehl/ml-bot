"""
kalshi_client.py
================
Standalone Kalshi API client for production_bot_v7.

API version: Kalshi Trade API v2 (post March 12, 2026 migration)
  - All prices sent as fixed-point dollar strings via yes_price_dollars / no_price_dollars
  - All counts sent as fixed-point strings via count_fp
  - No market order type (limit only)
  - post_only flag supported
  - time_in_force: "good_till_canceled" | "immediate_or_cancel" | "fill_or_kill"

New in v7 client:
  - create_order() accepts keyword args (yes_price, no_price, post_only, time_in_force,
    client_order_id) cleanly — bot passes these as **kwargs
  - amend_order() — reprice a resting order in-place
  - batch_create_orders() — submit multiple orders in one write call
  - get_orders() — list orders by status (used for startup reconciliation)
  - get_positions() — list open positions (used for startup reconciliation)
  - get_ws_auth_headers() — returns signed headers for Kalshi WebSocket connection
"""

import asyncio
import base64
import json
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiClient:
    """
    Async Kalshi API client. All methods are coroutines.

    Credentials (in priority order):
      1. Constructor arguments
      2. Environment variables: KALSHI_API_KEY, KALSHI_PRIVATE_KEY_PATH
    """

    BASE_URL = "https://api.elections.kalshi.com"
    WS_URL   = "wss://api.elections.kalshi.com/trade-api/ws/v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        private_key_path: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("KALSHI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing API key. Set KALSHI_API_KEY env var or pass api_key=.")

        key_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH")
        if not key_path:
            raise ValueError("Missing private key path. Set KALSHI_PRIVATE_KEY_PATH env var or pass private_key_path=.")

        self.private_key = self._load_private_key(key_path)
        # Timeout reduced from 30s to 6s (Fix #5).
        # Kalshi REST latency is typically 50-200ms. A 30s timeout would freeze
        # the tick loop (and stop checks) for up to 30s on a hung call.
        # 6s is generous for real slowdowns while bounding worst-case blindness.
        self.client = httpx.AsyncClient(timeout=6.0)
        print("✅ Kalshi Client Initialized")

    # ── Key loading ───────────────────────────────────────────────────────────

    def _load_private_key(self, path_str: str):
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Private key not found: {path}")
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)

    # ── Request signing ───────────────────────────────────────────────────────

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        """RSA-PSS signature over timestamp_ms + METHOD + /path"""
        msg = (timestamp_ms + method.upper() + path).encode("utf-8")
        sig = self.private_key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256().digest_size,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, path),
        }

    # ── WebSocket auth headers ────────────────────────────────────────────────

    async def get_ws_auth_headers(self) -> Dict[str, str]:
        """
        Returns signed headers for establishing an authenticated Kalshi WebSocket connection.
        The WS handshake path is /trade-api/ws/v2.
        Usage:
            headers = await kalshi.get_ws_auth_headers()
            async with websockets.connect(Config.KALSHI_WS_URL, extra_headers=headers) as ws:
                ...
        """
        path = "/trade-api/ws/v2"
        ts   = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY":       self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, "GET", path),
        }

    # ── Core HTTP request with retry ──────────────────────────────────────────

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        max_retries: int = 4,
    ) -> Dict:
        url     = self.BASE_URL + endpoint
        headers = self._auth_headers(method, endpoint)

        if params:
            url += "?" + urlencode({k: v for k, v in params.items() if v is not None})

        body = json.dumps(data) if data is not None else None

        for attempt in range(max_retries + 1):
            try:
                resp = await self.client.request(method, url, headers=headers, content=body)
                if resp.status_code == 429:
                    await asyncio.sleep(0.5 * (2 ** attempt) + random.random())
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    # Duplicate client_order_id — return the response body so
                    # caller can reconcile the existing order
                    try:
                        return e.response.json()
                    except Exception:
                        pass
                if e.response.status_code in (404, 410):
                    # Order/resource not found — no point retrying, raise immediately.
                    # Fast-failing here is critical: without this, a 404 on amend/cancel
                    # burns 7.5-11.5s in retry backoff before the bot's recovery logic
                    # fires, during which tick_loop can pile up redundant amend attempts
                    # against the same dead order.
                    raise
                if attempt == max_retries:
                    print(f"❌ API Error [{method} {endpoint}]: {e}")
                    raise
                await asyncio.sleep(0.5 * (2 ** attempt) + random.random())
            except Exception as e:
                if attempt == max_retries:
                    print(f"❌ Request Error [{method} {endpoint}]: {e}")
                    raise
                await asyncio.sleep(0.5 * (2 ** attempt) + random.random())
        return {}

    # ── Price / count helpers ─────────────────────────────────────────────────

    @staticmethod
    def _dollars_str(cents: int) -> str:
        """63 → '0.63'"""
        return f"{cents / 100:.2f}"

    @staticmethod
    def _count_str(count: int) -> str:
        """5 → '5.00'"""
        return f"{count:.2f}"

    # ── Portfolio ─────────────────────────────────────────────────────────────

    async def get_balance(self) -> Dict:
        return await self._request("GET", "/trade-api/v2/portfolio/balance")

    async def get_positions(self, **kwargs) -> Dict:
        """
        Get open positions.
        kwargs: series_ticker, ticker, etc.
        """
        return await self._request("GET", "/trade-api/v2/portfolio/positions", params=kwargs)

    # ── Markets ───────────────────────────────────────────────────────────────

    async def get_markets(self, max_retries: int = 4, **kwargs) -> Dict:
        return await self._request("GET", "/trade-api/v2/markets", params=kwargs,
                                   max_retries=max_retries)

    async def get_market(self, ticker: str) -> Dict:
        return await self._request("GET", f"/trade-api/v2/markets/{ticker}")

    async def get_orderbook(self, ticker: str, depth: int = 25, max_retries: int = 4) -> Dict:
        return await self._request(
            "GET", f"/trade-api/v2/markets/{ticker}/orderbook", params={"depth": depth},
            max_retries=max_retries,
        )

    # ── Orders ────────────────────────────────────────────────────────────────

    async def get_orders(self, **kwargs) -> Dict:
        """
        Get orders, optionally filtered.
        kwargs: status ("resting"|"filled"|"canceled"), ticker, etc.
        Used by startup_reconcile() to detect ghost orders from prior runs.
        """
        return await self._request("GET", "/trade-api/v2/portfolio/orders", params=kwargs)

    async def get_order(self, order_id: str) -> Dict:
        """
        Fetch a single order by ID.
        Used by _bg_settle() to verify fill_count > 0 before crediting payout.
        Response includes: order_id, status, fill_count, remaining_count, etc.
        """
        return await self._request("GET", f"/trade-api/v2/portfolio/orders/{order_id}")

    async def get_fills(self, **kwargs) -> Dict:
        """
        Fetch fills for the portfolio.
        kwargs: order_id, ticker, min_ts, max_ts, limit, cursor
        Used by _bg_settle() and _confirm_stop_exit() to get actual
        fill prices, costs, and fees from Kalshi.
        """
        return await self._request("GET", "/trade-api/v2/portfolio/fills", params=kwargs)

    async def create_order(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        client_order_id: Optional[str] = None,
        post_only: bool = False,
        time_in_force: str = "good_till_canceled",
    ) -> Dict:
        """
        Place a limit order on Kalshi.

        Args:
            ticker:           Market ticker.
            action:           "buy" or "sell".
            side:             "yes" or "no".
            count:            Number of contracts (integer).
            yes_price:        Price in cents for YES side (e.g. 63 = $0.63).
            no_price:         Price in cents for NO side. Provide one of yes_price/no_price.
            client_order_id:  Deterministic idempotency key. Auto-generated if omitted.
            post_only:        If True, Kalshi rejects the order rather than filling it as
                              a taker. Guarantees maker-only execution.
            time_in_force:    "good_till_canceled" (default, rests on book)
                              "immediate_or_cancel" (fill now or cancel remainder)
                              "fill_or_kill" (fill entire qty now or cancel entirely)

        Prices are sent as fixed-point dollar strings per Kalshi's March 12 2026 API migration.
        Counts are sent as fixed-point strings.
        """
        if yes_price is None and no_price is None:
            raise ValueError("Must provide yes_price or no_price")

        payload: Dict[str, Any] = {
            "ticker":           ticker,
            "action":           action,
            "side":             side,
            "count_fp":         self._count_str(count),
            "time_in_force":    time_in_force,
            "client_order_id":  client_order_id or str(uuid.uuid4()),
        }

        if yes_price is not None:
            payload["yes_price_dollars"] = self._dollars_str(yes_price)
        if no_price is not None:
            payload["no_price_dollars"]  = self._dollars_str(no_price)
        if post_only:
            payload["post_only"] = True

        return await self._request("POST", "/trade-api/v2/portfolio/orders", data=payload)

    async def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel a resting order.
        Used by _manage_active_order() for repricing (cancel → amend) and taker escalation.
        """
        return await self._request("DELETE", f"/trade-api/v2/portfolio/orders/{order_id}")

    async def amend_order(
        self,
        order_id: str,
        ticker: str,
        side: str,
        action: str,
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        count: Optional[int] = None,
        updated_client_order_id: Optional[str] = None,
    ) -> Dict:
        """
        Amend a resting order's price and/or quantity in-place.
        Preserves queue position (unlike cancel + resubmit).
        Exactly one of yes_price / no_price must be provided.

        Used by _manage_active_order() when bid drifts >= REPRICE_THRESHOLD.

        Args:
            order_id:                 ID of the order to amend.
            ticker:                   Market ticker (required by Kalshi amend endpoint).
            side:                     "yes" or "no".
            action:                   "buy" or "sell".
            yes_price:                New YES price in cents.
            no_price:                 New NO price in cents.
            count:                    New quantity (optional, omit to keep current).
            updated_client_order_id:  New client_order_id after amendment (optional).
        """
        if yes_price is None and no_price is None:
            raise ValueError("amend_order requires yes_price or no_price")

        payload: Dict[str, Any] = {
            "ticker": ticker,
            "side":   side,
            "action": action,
        }

        if yes_price is not None:
            payload["yes_price_dollars"] = self._dollars_str(yes_price)
        if no_price is not None:
            payload["no_price_dollars"]  = self._dollars_str(no_price)
        if count is not None:
            payload["count_fp"] = self._count_str(count)
        if updated_client_order_id:
            payload["updated_client_order_id"] = updated_client_order_id

        return await self._request(
            "POST", f"/trade-api/v2/portfolio/orders/{order_id}/amend", data=payload
        )

    async def batch_create_orders(self, orders: List[Dict]) -> Dict:
        """
        Submit multiple orders in a single API call.
        Each order in the batch counts as 1 write transaction against rate limit.

        Used for:
          - Large entry ladders: multiple Maker GtC orders across book depth.
          - Large stop exit sweeps: multiple IoC sell orders across book depth simultaneously.

        Each dict in `orders` should have:
            ticker, action, side, count, yes_price OR no_price,
            time_in_force (optional), post_only (optional), client_order_id (optional)

        Internally converts count (int) → count_fp and price (int cents) → dollars string.
        Returns the raw Kalshi response with an "orders" list.
        """
        converted = []
        for o in orders:
            order: Dict[str, Any] = {
                "ticker":          o["ticker"],
                "action":          o["action"],
                "side":            o["side"],
                "count_fp":        self._count_str(int(o["count"])),
                "time_in_force":   o.get("time_in_force", "good_till_canceled"),
                "client_order_id": o.get("client_order_id") or str(uuid.uuid4()),
            }
            if "yes_price" in o and o["yes_price"] is not None:
                order["yes_price_dollars"] = self._dollars_str(int(o["yes_price"]))
            if "no_price" in o and o["no_price"] is not None:
                order["no_price_dollars"]  = self._dollars_str(int(o["no_price"]))
            if o.get("post_only"):
                order["post_only"] = True

            converted.append(order)

        return await self._request(
            "POST",
            "/trade-api/v2/portfolio/orders/batched",
            data={"orders": converted},
        )
