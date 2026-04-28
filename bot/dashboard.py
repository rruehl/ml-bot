"""
dashboard_v8_ml.py — v8.0.0
============================
Flask dashboard for production_bot_v8_ml (v8.0.0).

Changes from v7.6.0 (v7.6.2):
  - FIX: avg_entry now uses only FILL_CONFIRMED rows where kalshi_fill_qty > 0
    (actual fills), not all ORDER_RESTING rows. Previously, cancelled/repriced
    attempts were included, skewing the average. Falls back to ORDER_RESTING
    for paper mode or sessions with no fill rows yet.
  - FIX: avg_signal_age now derives from the same actual-fill rows as avg_entry,
    so it reflects the age at the real fill rather than all order attempts.
  - FIX: avg_slippage now merges FILL_VERIFIED (REST-confirmed price) against
    the matching FILL_CONFIRMED (posted price) by order_id, de-duplicating the
    WS+REST double-confirmation pattern so each fill is counted exactly once.

Changes from v7.5 (v7.6):
  - ADD: Activity log now shows YES/NO settlement outcome for every session,
         including sessions where no position was held. Previously, no-position
         sessions showed a generic "Market roll" entry with no outcome. Now they
         show "SETTLED YES" (green) or "SETTLED NO" (red), matching the color
         convention of wins/losses. Requires production_bot_v7.py v7.6.0 which
         adds _bg_settle_no_position() to fetch the outcome from Kalshi for every
         session roll, even with no position. Old-format CSV rows (pre-v7.6) show
         "SETTLED" in blue as before so historical logs are not broken.

Changes from v7.4 (v7.5):
  - ADD: TAKER_ESCALATION, LADDER_PARTIAL, STOP_PARTIAL event colors and
         show_events entries.

Bug fixes from v7.1 (carried forward):
  - FIX: Win/loss counting no longer double-counts stops. STOP_EXIT + STOP_CONFIRMED
         both fire for one live stop, so only STOP_CONFIRMED is counted as a loss
         closure. Paper mode STOP_EXIT is counted directly (no STOP_CONFIRMED in paper).
         SETTLE_VERIFIED (market rolls with no position) are excluded entirely.
  - FIX: Live balance fallback — when Kalshi API call fails, dashboard now falls back
         to the most recent BALANCE_SYNC bankroll value from the CSV rather than
         showing $0.00 and -100% ROI. Source indicator shown in the UI.
  - FIX: fetch_live_balance() now correctly parses the API response. The Kalshi
         balance endpoint returns {"balance": <cents int>} — there is no
         "balance_dollars" field. The dead first branch has been removed.
  - FIX: ob_stale flag now uses only the single most recent CSV row, not a
         majority vote over the last 5 rows (which kept showing stale after recovery).
  - FIX: Active position detection uses STOP_CONFIRMED (not STOP_EXIT) as the
         position close event so the card doesn't flicker between the two events.
  - FIX: PnL chart "start" point now anchors to the bot's first log timestamp
         rather than the first trade timestamp, giving accurate time context.
  - FIX: avg_spread removed from template context (it was computed but unused and
         the column reference was fragile).

Data sources:
  - production_log_v8.csv  — all bot events, Kalshi-confirmed fills, balance syncs
  - Kalshi REST API        — live balance fetched on each page load (fallback to CSV)
  - config.json            — live config values displayed in panel
"""

import glob
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from flask import Flask, jsonify, render_template_string

import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from kalshi_client import KalshiClient
    import asyncio
    _kalshi = KalshiClient()
    _kalshi_ok = True
except Exception as _e:
    print(f"[DASHBOARD] WARNING: Could not init KalshiClient: {_e}")
    _kalshi = None
    _kalshi_ok = False

app = Flask(__name__)

CSV_FILE = str(Path(__file__).resolve().parent.parent / "logs" / "production_log.csv")

EVENT_COLORS = {
    "PAPER_BUY":         "#00e676",
    "LIVE_BUY":          "#00e676",
    "ORDER_RESTING":     "#00e676",
    "FILL_CONFIRMED":    "#00e676",
    "FILL_VERIFIED":     "#00e676",
    "ORDER_AMENDED":     "#ffd740",
    "ORDER_ESCALATED":   "#ff9800",
    "TAKER_ESCALATION":  "#ff9800",  # v7.4.0: taker fee warning (orange)
    "LADDER_PARTIAL":    "#ffd740",  # v7.4.0: partial fill warning (yellow)
    "STOP_PARTIAL":      "#ff9800",  # v7.4.0: orphaned contracts warning (orange)
    "STOP_FAILED_EXPIRY":"#ff5252",  # v7.7.0: stop failed, position expired (real PnL record)
    "STOP_EXPIRY_RISK":  "#ff9800",  # v7.7.1: real-time alert — stop IoCs exhausted, no PnL yet
    "RECONCILE":         "#cc44ff",  # v7.7.0: post-settlement Kalshi reconciliation
    "CIRCUIT_BREAKER":   "#ff0000",  # v7.5.0: daily loss limit hit (bright red)
    "ML_INFERENCE":      "#00e5ff",  # v8.0: ML inference fired (cyan)
    "MODEL_RELOAD":      "#cc44ff",  # v8.0: weekly model hot-reload
    "ORDER_UNFILLED":    "#ff5252",
    "PAYOUT":            "#00e676",
    "STOP_EXIT":         "#ff5252",
    "STOP_CONFIRMED":    "#ff5252",
    "STOP_FAILED":       "#ff0000",
    "SETTLE_VERIFIED":   "#448aff",
    "SETTLE":            "#ff5252",
    "BALANCE_SYNC":      "#448aff",
    "ERROR":             "#ff5252",
}

def _event_color(event: str) -> str:
    for k, c in EVENT_COLORS.items():
        if k in event:
            return c
    return "#888"

def safe_float(val, default=0.0):
    try:    return float(val)
    except: return default

def safe_int(val, default=0):
    try:    return int(val)
    except: return default


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta id="refresh-meta" http-equiv="refresh" content="10">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <title>Edge Bot 5min</title>
    <style>
        :root {
            --bg:#0d0d0d; --card:#141414; --border:#222; --text:#e0e0e0;
            --muted:#555; --green:#00e676; --red:#ff5252; --yellow:#ffd740;
            --orange:#ff9800; --blue:#448aff; --purple:#cc44ff;
        }
        *{box-sizing:border-box;margin:0;padding:0;}
        body{font-family:-apple-system,system-ui,sans-serif;background:var(--bg);color:var(--text);padding:10px;font-size:13px;}
        .card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px;margin-bottom:10px;}
        .card-title{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;}
        .big-val{font-size:1.7rem;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.1;}
        .med-val{font-size:1.1rem;font-weight:700;font-variant-numeric:tabular-nums;}
        .sub{font-size:.75rem;color:var(--muted);margin-top:3px;line-height:1.8;}
        .green{color:var(--green);} .red{color:var(--red);} .yellow{color:var(--yellow);}
        .blue{color:var(--blue);} .orange{color:var(--orange);}
        .header{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:var(--card);border:1px solid var(--border);border-radius:8px;margin-bottom:10px;}
        .header-title{font-size:.9rem;font-weight:700;letter-spacing:2px;}
        .header-right{text-align:right;font-size:.7rem;color:var(--muted);line-height:1.6;}
        .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:middle;}
        .dot-green{background:var(--green);box-shadow:0 0 6px var(--green);}
        .dot-yellow{background:var(--yellow);box-shadow:0 0 4px var(--yellow);}
        .dot-red{background:var(--red);}
        .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;}
        .grid-4{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;margin-bottom:10px;}
        .sig-pill{display:inline-block;padding:2px 10px;border-radius:4px;font-size:.75rem;font-weight:700;letter-spacing:1px;}
        .sig-buy{background:rgba(0,230,118,.15);color:var(--green);border:1px solid var(--green);}
        .sig-sell{background:rgba(255,82,82,.15);color:var(--red);border:1px solid var(--red);}
        .sig-idle{background:rgba(255,215,64,.15);color:var(--yellow);border:1px solid var(--yellow);}
        .pill{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.7rem;font-weight:700;}
        .pill-green{background:rgba(0,230,118,.15);color:var(--green);border:1px solid var(--green);}
        .pill-yellow{background:rgba(255,215,64,.15);color:var(--yellow);border:1px solid var(--yellow);}
        .hbar{height:3px;background:var(--border);border-radius:2px;margin-top:6px;}
        .hbar-fill{height:100%;border-radius:2px;transition:width .3s;}
        .chart-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}
        .chart-btns{display:flex;gap:4px;}
        .btn-t{background:var(--border);border:1px solid #333;color:var(--muted);padding:3px 7px;border-radius:4px;cursor:pointer;font-size:.7rem;font-family:inherit;}
        .btn-t.active{background:var(--green);color:#000;font-weight:700;border-color:var(--green);}
        .chart-wrap{position:relative;height:300px;}
        table{width:100%;border-collapse:collapse;font-size:.78rem;}
        th{text-align:left;color:var(--muted);border-bottom:1px solid var(--border);padding:5px 0;font-size:.65rem;text-transform:uppercase;letter-spacing:.8px;}
        td{padding:5px 0;border-bottom:1px solid #1a1a1a;vertical-align:middle;}
        td.msg-col{color:var(--muted);font-size:.72rem;word-break:break-word;max-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
        .cfg-table td{padding:3px 6px;font-size:.72rem;font-family:monospace;}
        .cfg-table td:first-child{color:var(--muted);width:55%;}
        .cfg-table td:last-child{color:var(--blue);}
        .section-label{font-size:.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin:10px 0 6px;border-top:1px solid var(--border);padding-top:8px;}
        details summary{cursor:pointer;color:var(--muted);font-size:.75rem;padding:4px 0;}
        details summary:hover{color:var(--text);}
        .toggle-btns{display:flex;gap:4px;margin-bottom:8px;}
        .t-btn{background:var(--border);border:1px solid #333;color:var(--muted);padding:3px 8px;border-radius:4px;cursor:pointer;font-size:.7rem;font-family:inherit;}
        .t-btn.active{background:var(--blue);color:#fff;font-weight:700;border-color:var(--blue);}
        .pause-btn{background:var(--border);border:1px solid #333;color:var(--muted);padding:4px 10px;border-radius:4px;cursor:pointer;font-size:.7rem;font-family:inherit;letter-spacing:.5px;}
        .pause-btn.paused{background:rgba(255,215,64,.15);color:var(--yellow);border-color:var(--yellow);font-weight:700;}
    </style>
</head>
<body>

<!-- Header -->
<div class="header">
    <div>
        <div class="header-title">
            <span class="dot {{ 'dot-green' if is_active and ws_connected else 'dot-yellow' if is_active else 'dot-red' }}"></span>
            EDGE BOT <span style="color:#333;">v8.0-ml</span>
        </div>
        <div style="font-size:.65rem;color:var(--muted);margin-top:2px;">
            {{ mode }} &nbsp;·&nbsp; {{ 'Online' if is_active else 'Offline' }}
            &nbsp;·&nbsp; <span style="color:var(--blue);">Runtime τ:{{ "{:.3f}".format(ml_tau) }}</span>
            &nbsp;·&nbsp; <span style="color:var(--blue);">ATR:{{ atr_min|int }}–{{ atr_max|int }}</span>
            {% if not ws_connected %}<span style="color:var(--red);"> · WS DISCONNECTED</span>{% endif %}
            {% if ob_stale %}<span style="color:var(--yellow);"> · Stale OB ({{ ob_delta_age }}s)</span>{% endif %}
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
        <button id="pauseBtn" class="pause-btn" onclick="togglePause()">⏸ Pause</button>
        <div class="header-right">
            {{ last_update }} CST<br>
            <span id="refresh-status" style="color:#555;">↻ <span id="countdown">10</span>s</span>
        </div>
    </div>
</div>

<!-- Balance + PnL -->
<div class="grid-2">
    <div class="card">
        <div class="card-title">Live Balance
            <span style="font-size:.6rem;color:var(--blue);">↻ Kalshi API</span>
        </div>
        <div class="big-val green">${{ "{:,.2f}".format(live_balance) }}</div>
        <div class="sub">
            ROI: <span class="{{ 'green' if roi >= 0 else 'red' }}">{{ "{:+.1f}".format(roi) }}%</span>
            &nbsp;·&nbsp; Deposit: ${{ "{:,.2f}".format(starting_deposit) }}
        </div>
        <div class="sub">
            Fees paid: <span class="red">${{ "{:.2f}".format(total_fees) }}</span>
            &nbsp;·&nbsp; 24h loss: <span class="red">${{ "{:.2f}".format(rolling_loss_24h) }}</span>
        </div>
        <div class="sub">
            Active Kelly Risk: <span class="blue">{{ "{:.2f}".format(active_risk_pct) }}%</span> per trade
        </div>
    </div>
    <div class="card">
        <div class="card-title">Realized P&amp;L
            {% if reconcile_count > 0 %}
            <span style="font-size:.6rem;color:var(--purple);">✓ Kalshi reconciled ({{ reconcile_count }} trades)</span>
            {% else %}
            <span style="font-size:.6rem;color:var(--green);">from Kalshi fills</span>
            {% endif %}
        </div>
        <div class="big-val {{ 'green' if realized_pnl >= 0 else 'red' }}">${{ "{:+.2f}".format(realized_pnl) }}</div>
        {% if reconcile_divergence %}
        <div style="margin-top:4px;">
            <span class="pill" style="background:rgba(255,152,0,.15);color:var(--orange);border:1px solid var(--orange);">
                ⚠ {{ reconcile_divergence_count }} divergence{{ 's' if reconcile_divergence_count != 1 else '' }} &gt;$0.10
            </span>
        </div>
        {% endif %}
        <div class="sub">
            Gross proceeds: <span class="green">${{ "{:.2f}".format(gross_proceeds_total) }}</span><br>
            Gross cost: <span class="red">${{ "{:.2f}".format(gross_cost_total) }}</span><br>
            Net fees: <span class="red">${{ "{:.2f}".format(total_fees) }}</span>
        </div>
    </div>
</div>

<!-- Quick Stats + Market -->
<div class="grid-2">
    <div class="card">
        <div class="card-title">Performance</div>
        <div class="sub" style="line-height:2.2;">
            Win Rate: <span class="{{ 'green' if win_rate >= 55 else 'yellow' if win_rate >= 50 else 'red' }}">{{ "{:.1f}".format(win_rate) }}%</span>
            &nbsp;·&nbsp; <span class="blue">{{ wins }}W / {{ losses }}L</span><br>
            ROI: <span class="{{ 'green' if roi >= 0 else 'red' }}">{{ "{:+.2f}".format(roi) }}%</span>
            &nbsp;·&nbsp; P&amp;L: <span class="{{ 'green' if realized_pnl >= 0 else 'red' }}">${{ "{:+.2f}".format(realized_pnl) }}</span><br>
            Drift (L50): <span class="{{ 'green' if drift_win_rate >= 55 else 'orange' if drift_win_rate >= 50 else 'red' }}">{{ "{:.1f}".format(drift_win_rate) }}%</span>
            &nbsp;·&nbsp; Kelly risk: <span class="blue">{{ "{:.2f}".format(active_risk_pct) }}%</span>
        </div>
    </div>
    <div class="card">
        <div class="card-title">Market</div>
        <div class="sub" style="line-height:2.2;">
            <span style="color:var(--text);font-size:.8rem;">{{ ticker }}</span><br>
            BTC: <span class="yellow">${{ "{:,.2f}".format(btc_price) }}</span>
            &nbsp;·&nbsp; Strike: ${{ "{:,.0f}".format(strike) }}<br>
            Expires: <span class="{{ 'red' if time_left < 3 else 'yellow' if time_left < 7 else '' }}">{{ "{:.1f}".format(time_left) }}m</span>
            &nbsp;·&nbsp; ATR: <span class="{{ 'green' if atr_min <= live_atr <= atr_max else 'red' }}">{{ "{:.2f}".format(live_atr) }}</span> <span style="color:var(--muted);font-size:.68rem;">(gate {{ atr_min|int }}–{{ atr_max|int }})</span>
        </div>
    </div>
</div>

<!-- Active position / order card -->
{% if has_active_position or has_active_order %}
<div class="card" style="border-color:{{ '#00e676' if has_active_position else '#ffd740' }};">
    <div class="card-title">{{ 'Open Position' if has_active_position else 'Resting Order' }}</div>
    {% if has_active_position %}
    <div class="sub" style="line-height:2.0;">
        <span class="pill pill-green">IN POSITION</span>&nbsp;
        {{ pos_ticker }} &nbsp;·&nbsp; {{ pos_side | upper }} @ <span class="green">{{ pos_entry }}¢</span> x{{ pos_qty }}<br>
        Confirmed fill: <span class="green">{{ pos_fill_price }}¢ x{{ pos_fill_qty }}</span>
        &nbsp;·&nbsp; Cost: <span class="red">${{ pos_cost }}</span>
        &nbsp;·&nbsp; Fees: <span class="red">${{ pos_fees }}</span><br>
        Trail: <span class="yellow">{{ stop_trail }}¢</span>
        &nbsp;·&nbsp; Best bid: <span class="green">{{ stop_best_bid }}¢</span>
        &nbsp;·&nbsp; Stop level: <span class="red">{{ stop_level }}¢</span>
        &nbsp;·&nbsp; Active: <span class="{{ 'green' if stop_active else 'yellow' }}">{{ 'YES' if stop_active else 'PENDING' }}</span>
    </div>
    {% else %}
    <div class="sub" style="line-height:2.0;">
        <span class="pill pill-yellow">RESTING</span>&nbsp;
        {{ order_ticker }} &nbsp;·&nbsp; {{ order_side | upper }} @ <span class="yellow">{{ order_price }}¢</span> x{{ order_qty }}<br>
        Age: {{ order_age }}s &nbsp;·&nbsp; Amends: {{ order_amends }}<br>
        <span style="font-family:monospace;font-size:.7rem;color:var(--muted);">{{ order_coid }}</span>
    </div>
    {% endif %}
</div>
{% endif %}

<!-- Session Inference History -->
<div class="card">
    <div class="card-title">Session Inference History <span style="color:var(--muted);font-size:.6rem;">(last 10 sessions)</span></div>
    <table>
        <thead>
            <tr>
                <th style="width:16%;">Time</th>
                <th style="width:14%;">Prediction</th>
                <th style="width:18%;">Conf / τ</th>
                <th style="width:24%;">Status</th>
                <th>Outcome</th>
            </tr>
        </thead>
        <tbody>
            {% for row in inference_history %}
            <tr>
                <td style="color:var(--muted);">{{ row.time }}</td>
                <td>
                    <span class="{{ 'green' if row.direction == 'UP' else 'red' if row.direction == 'DN' else '' }}">
                        {{ row.direction }}
                    </span>
                </td>
                <td>
                    <span class="{{ 'green' if row.confidence >= row.effective_tau else 'red' }}">
                        {{ "{:.1f}".format(row.confidence) }}%
                    </span>
                    <span style="color:var(--muted);font-size:.68rem;">&nbsp;/&nbsp;{{ "{:.1f}".format(row.effective_tau) }}%</span>
                </td>
                <td>
                    {% if row.status == 'success' %}
                    <span class="green">✓ traded</span>
                    {% elif row.status == 'tau_gate_fail' %}
                    <span class="red">τ gate fail</span>
                    {% else %}
                    <span class="orange">filtered</span>
                    {% endif %}
                </td>
                <td>
                    {% if row.outcome == 'correct' %}
                    <span class="green">✓ correct</span>
                    {% elif row.outcome == 'wrong' %}
                    <span class="red">✗ wrong</span>
                    {% elif row.outcome == 'pending' %}
                    <span style="color:var(--muted);">pending</span>
                    {% else %}
                    <span style="color:var(--muted);">–</span>
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
            {% if not inference_history %}
            <tr><td colspan="5" style="text-align:center;padding:12px;color:var(--muted);">No ML inferences yet</td></tr>
            {% endif %}
        </tbody>
    </table>
</div>

<!-- ML Signal Card -->
<div class="card" style="border-color:{{ '#00e676' if ml_direction == 1 else '#ff5252' if ml_direction == 0 else '#333' }};">
    <div class="card-title">ML Signal &nbsp;
        <span style="font-size:.6rem;color:var(--muted);">two_class_model &nbsp;·&nbsp;
            runtime τ=<span style="color:var(--blue);">{{ "{:.3f}".format(ml_tau) }}</span>
            &nbsp;·&nbsp; trained τ=<span style="color:var(--muted);">{{ "{:.3f}".format(trained_tau) }}</span>
        </span>
    </div>
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        {% if ml_direction == 1 %}
        <span class="sig-pill sig-buy">&#9650; UP / YES</span>
        {% elif ml_direction == 0 %}
        <span class="sig-pill sig-sell">&#9660; DOWN / NO</span>
        {% else %}
        <span class="sig-pill sig-idle">-- NO SIGNAL --</span>
        {% endif %}
        <div>
            <div class="sub">
                Confidence: <span class="{{ 'green' if ml_confidence >= ml_tau else 'yellow' if ml_confidence > 0 else 'muted' }}">
                    {{ "{:.1f}".format(ml_confidence * 100) }}%
                </span>
                &nbsp;·&nbsp; proba_up: <span class="blue">{{ "{:.4f}".format(ml_proba_up) }}</span>
            </div>
            <div class="sub">
                Strike: <span class="yellow">${{ "{:,.0f}".format(strike) }}</span>
                &nbsp;·&nbsp; Spot: <span class="yellow">${{ "{:,.2f}".format(btc_price) }}</span>
                &nbsp;·&nbsp; Moneyness:
                <span class="{{ 'green' if moneyness_bps <= 0 else 'yellow' if moneyness_bps <= 20 else 'red' }}">
                    {{ "{:+.1f}".format(moneyness_bps) }} bps
                </span>
            </div>
            <div class="sub">
                Signal age: {{ ('{:.1f}m'.format(ml_signal_age)) if ml_signal_age is not none else '--' }}
                &nbsp;·&nbsp; Fired: {{ ml_fired_at }}
            </div>
        </div>
    </div>
    {% if ml_direction is not none and ml_confidence > 0 %}
    <div class="hbar" style="margin-top:8px;">
        <div class="hbar-fill" style="width:{{ [ml_confidence * 100, 100] | min }}%;
            background:{{ 'var(--green)' if ml_confidence >= ml_tau else 'var(--yellow)' }};"></div>
    </div>
    {% endif %}
</div>

<!-- Win Rate + Trade Quality -->
<div class="grid-2">
    <div class="card">
        <div class="card-title">Win Rate</div>
        <div class="big-val {{ 'green' if win_rate >= 55 else 'yellow' if win_rate >= 50 else 'red' }}">{{ "{:.1f}".format(win_rate) }}%</div>
        <div class="sub">{{ wins }}W / {{ losses }}L / {{ total_trades }} settled</div>
        <div class="hbar"><div class="hbar-fill" style="width:{{ win_rate }}%;background:{{ 'var(--green)' if win_rate >= 55 else 'var(--yellow)' if win_rate >= 50 else 'var(--red)' }};"></div></div>
        <div class="sub" style="margin-top:4px;">Drift (last 50): <span class="{{ 'green' if drift_win_rate >= 55 else 'orange' if drift_win_rate >= 50 else 'red' }}">{{ "{:.1f}".format(drift_win_rate) }}%</span></div>
    </div>
    <div class="card">
        <div class="card-title">Trade Quality</div>
        <div class="sub">
            Avg entry: <span class="blue">{{ avg_entry }}¢</span>
            &nbsp;·&nbsp; Avg slippage: <span class="{{ 'green' if avg_slippage <= 0 else 'red' }}">{{ avg_slippage }}¢</span><br>
            Avg signal age: {{ avg_signal_age }}m &nbsp;·&nbsp;
                Avg ML conf: <span class="blue">{{ avg_ml_confidence }}%</span><br>
            Avg PnL: <span class="{{ 'green' if avg_pnl >= 0 else 'red' }}">${{ "{:+.2f}".format(avg_pnl) }}</span>
            &nbsp;·&nbsp; Avg fees: <span class="red">${{ "{:.2f}".format(avg_fees) }}</span>
        </div>
    </div>
</div>

<div class="grid-2">
    <div class="card">
        <div class="card-title">Collector
            <span class="dot {{ 'dot-red' if collector_stale else 'dot-green' }}" style="margin-left:6px;"></span>
        </div>
        <div class="sub">
            Last bar: <span class="blue">{{ collector_last_ts }}</span><br>
            Bars today: <span class="blue">{{ collector_rows_today }}</span>
            {% if collector_stale %}<span style="color:var(--red);"> · STALE &gt;3 min</span>{% endif %}
        </div>
    </div>
    <div class="card">
        <div class="card-title">Model</div>
        <div class="sub" style="line-height:2.1;">
            Trained: <span class="blue">{{ model_trained_at }}</span><br>
            Acc: <span class="blue">{{ model_dir_acc }}%</span>
            &nbsp;·&nbsp; Win rate: <span class="blue">{{ model_win_rate }}%</span>
            &nbsp;·&nbsp; Coverage: <span class="{{ 'red' if model_atr_filtered else 'blue' }}">{{ model_coverage }}%</span>
            {% if model_atr_filtered %}<span style="color:var(--red);font-size:.65rem;"> ⚠ ATR-filtered</span>{% endif %}<br>
            Trained τ: <span class="blue">{{ model_tau }}</span>
            &nbsp;·&nbsp; Runtime τ: <span class="green">{{ "{:.3f}".format(ml_tau) }}</span><br>
            Next retrain: <span style="color:var(--yellow);">{{ next_retrain }}</span>
        </div>
    </div>
</div>

<!-- Stats toggle: All / 24h / 7d -->
<div class="card">
    <div class="chart-header" style="margin-bottom:8px;">
        <div class="card-title" style="margin:0;">Performance Stats</div>
        <div class="toggle-btns">
            <button class="t-btn active" onclick="switchWindow('all',this)">All</button>
            <button class="t-btn" onclick="switchWindow('24h',this)">24h</button>
            <button class="t-btn" onclick="switchWindow('7d',this)">7d</button>
        </div>
    </div>
    <div id="stats-all">
        <div class="grid-4" style="margin:0;">
            <div><div class="card-title">Profit Factor</div>
                <div class="med-val {{ 'green' if pf_all >= 1.5 else 'yellow' if pf_all >= 1 else 'red' }}">{{ "{:.2f}".format(pf_all) if pf_all != 9999 else '∞' }}</div>
                <div class="sub">all time</div></div>
            <div><div class="card-title">Sharpe Ratio</div>
                <div class="med-val {{ 'green' if sharpe_all >= 1 else 'yellow' if sharpe_all >= 0 else 'red' }}">{{ "{:.2f}".format(sharpe_all) }}</div>
                <div class="sub">per-trade</div></div>
            <div><div class="card-title">Gross Win</div>
                <div class="med-val green">${{ "{:.4f}".format(gross_win_all) }}</div>
                <div class="sub">{{ wins_all }} trades</div></div>
            <div><div class="card-title">Gross Loss</div>
                <div class="med-val red">${{ "{:.4f}".format(gross_loss_all) }}</div>
                <div class="sub">{{ losses_all }} trades</div></div>
        </div>
    </div>
    <div id="stats-24h" style="display:none;">
        <div class="grid-4" style="margin:0;">
            <div><div class="card-title">Profit Factor</div>
                <div class="med-val {{ 'green' if pf_24h >= 1.5 else 'yellow' if pf_24h >= 1 else 'red' }}">{{ "{:.2f}".format(pf_24h) if pf_24h != 9999 else '∞' }}</div>
                <div class="sub">last 24h</div></div>
            <div><div class="card-title">Sharpe Ratio</div>
                <div class="med-val {{ 'green' if sharpe_24h >= 1 else 'yellow' if sharpe_24h >= 0 else 'red' }}">{{ "{:.2f}".format(sharpe_24h) }}</div>
                <div class="sub">per-trade</div></div>
            <div><div class="card-title">Gross Win</div>
                <div class="med-val green">${{ "{:.4f}".format(gross_win_24h) }}</div>
                <div class="sub">{{ wins_24h }} trades</div></div>
            <div><div class="card-title">Gross Loss</div>
                <div class="med-val red">${{ "{:.4f}".format(gross_loss_24h) }}</div>
                <div class="sub">{{ losses_24h }} trades</div></div>
        </div>
    </div>
    <div id="stats-7d" style="display:none;">
        <div class="grid-4" style="margin:0;">
            <div><div class="card-title">Profit Factor</div>
                <div class="med-val {{ 'green' if pf_7d >= 1.5 else 'yellow' if pf_7d >= 1 else 'red' }}">{{ "{:.2f}".format(pf_7d) if pf_7d != 9999 else '∞' }}</div>
                <div class="sub">last 7d</div></div>
            <div><div class="card-title">Sharpe Ratio</div>
                <div class="med-val {{ 'green' if sharpe_7d >= 1 else 'yellow' if sharpe_7d >= 0 else 'red' }}">{{ "{:.2f}".format(sharpe_7d) }}</div>
                <div class="sub">per-trade</div></div>
            <div><div class="card-title">Gross Win</div>
                <div class="med-val green">${{ "{:.4f}".format(gross_win_7d) }}</div>
                <div class="sub">{{ wins_7d }} trades</div></div>
            <div><div class="card-title">Gross Loss</div>
                <div class="med-val red">${{ "{:.4f}".format(gross_loss_7d) }}</div>
                <div class="sub">{{ losses_7d }} trades</div></div>
        </div>
    </div>
</div>

<!-- Orderbook -->
<div class="grid-2">
    <div class="card">
        <div class="card-title">Order Book &nbsp;<span style="font-size:.6rem;color:var(--muted);">WS delta {{ ob_delta_age }}s ago</span></div>
        <div class="sub">
            Yes bid: <span class="green">{{ yes_bid }}¢</span> ask: {{ yes_ask }}¢ &nbsp;·&nbsp; Liq: {{ yes_liq }}<br>
            No bid: <span class="red">{{ no_bid }}¢</span> ask: {{ no_ask }}¢ &nbsp;·&nbsp; Liq: {{ no_liq }}<br>
            OBI: <span class="{{ 'green' if obi > 0.2 else 'red' if obi < -0.2 else '' }}">{{ "{:+.3f}".format(obi) }}</span>
            <span style="color:var(--muted);font-size:.68rem;">&nbsp;({{ 'YES-heavy' if obi > 0.05 else 'NO-heavy' if obi < -0.05 else 'balanced' }})</span>
        </div>
    </div>
    <div class="card">
        <div class="card-title">WS Health</div>
        <div class="sub">
            Kalshi WS: <span class="{{ 'green' if ws_connected else 'red' }}">{{ 'Connected' if ws_connected else 'DISCONNECTED' }}</span><br>
            Last delta: <span class="{{ 'green' if ob_delta_age < 5 else 'yellow' if ob_delta_age < 30 else 'red' }}">{{ ob_delta_age }}s ago</span><br>
            OB stale: <span class="{{ 'green' if not ob_stale else 'red' }}">{{ 'No' if not ob_stale else 'YES' }}</span>
        </div>
    </div>
</div>

<!-- PnL Chart -->
<div class="card">
    <div class="chart-header">
        <div class="card-title" style="margin:0;">Realized P&amp;L Curve <span style="font-size:.6rem;color:var(--green);">actual Kalshi amounts</span></div>
        <div class="chart-btns">
            <button class="btn-t" onclick="filterChart(1,this)">1H</button>
            <button class="btn-t" onclick="filterChart(6,this)">6H</button>
            <button class="btn-t" onclick="filterChart(24,this)">24H</button>
            <button class="btn-t active" onclick="filterChart(0,this)">ALL</button>
        </div>
    </div>
    <div class="chart-wrap"><canvas id="pnlChart"></canvas></div>
</div>

<!-- Activity Log -->
<div class="card">
    <div class="card-title">Activity Log</div>
    <table>
        <thead>
            <tr>
                <th style="width:15%;">Time</th>
                <th style="width:22%;">Event</th>
                <th>Detail</th>
            </tr>
        </thead>
        <tbody>
            {% for row in logs %}
            <tr>
                <td style="color:var(--muted);">{{ row.time }}</td>
                <td><span style="color:{{ row.color }};font-size:.75rem;">{{ row.event }}</span></td>
                <td class="msg-col">{{ row.msg }}</td>
            </tr>
            {% endfor %}
            {% if not logs %}
            <tr><td colspan="3" style="text-align:center;padding:15px;color:var(--muted);">No activity yet</td></tr>
            {% endif %}
        </tbody>
    </table>
</div>

<!-- Config Panel -->
<div class="card">
    <details>
        <summary>⚙️ Live Config (config.json)</summary>
        <br>
        {% for section, items in config_sections.items() %}
        <div class="section-label">{{ section }}</div>
        <table class="cfg-table">
            {% for k, v in items %}
            <tr><td>{{ k }}</td><td>{{ v }}</td></tr>
            {% endfor %}
        </table>
        {% endfor %}
    </details>
</div>

<script>
const allLabels     = {{ chart_labels | tojson }};
const allData       = {{ chart_data | tojson }};
const allTimestamps = {{ chart_timestamps | tojson }};
const ctx = document.getElementById('pnlChart').getContext('2d');
let chart;
let activeChartHours = 0;  // 0 = ALL

// ── Restore persisted filter selections ───────────────────────────────────────
const _savedChart  = localStorage.getItem('edgebot_chart_window') || '0';
const _savedStats  = localStorage.getItem('edgebot_stats_window') || 'all';
activeChartHours   = parseInt(_savedChart);

function getGradient(cx, chartArea, scales) {
    if (!chartArea || !scales || !scales.y) return '#00e676';
    const zeroPx = scales.y.getPixelForValue(0);
    const t = chartArea.top, b = chartArea.bottom, h = b - t;
    if (h <= 0) return '#00e676';
    const zr = Math.max(0, Math.min(1, (zeroPx - t) / h));
    const g = cx.createLinearGradient(0, t, 0, b);
    g.addColorStop(0,'#00e676'); g.addColorStop(zr,'#00e676');
    g.addColorStop(zr,'#ff5252'); g.addColorStop(1,'#ff5252');
    return g;
}

function buildChart(labels, data, tradeIndices) {
    if (chart) chart.destroy();
    // Overlay dataset: dots at each trade point
    const tradePoints = data.map((v,i) => tradeIndices.includes(i) ? v : null);
    chart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets: [
            { data, borderWidth: 2,
              pointRadius: 0, pointHoverRadius: 4,
              fill: { target: 'origin', above: 'rgba(0,230,118,.07)', below: 'rgba(255,82,82,.07)' },
              tension: 0.3 },
            { data: tradePoints, borderWidth: 0,
              pointRadius: 5, pointHoverRadius: 6,
              pointStyle: 'circle',
              pointBackgroundColor: tradePoints.map(v => v === null ? 'transparent' : (v >= 0 ? '#00e676' : '#ff5252')),
              pointBorderColor: '#000',
              pointBorderWidth: 1,
              showLine: false, fill: false }
        ]},
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { intersect: false, mode: 'index' },
            scales: {
                x: { ticks: { color: '#444', maxTicksLimit: 6, font: { size: 10 } }, grid: { color: '#1a1a1a' } },
                y: { ticks: { color: '#444', font: { size: 10 }, callback: v => '$'+v.toFixed(4) }, grid: { color: '#1a1a1a' } }
            },
            plugins: { legend: { display: false },
                tooltip: { backgroundColor: '#1a1a1a', titleColor: '#888', bodyColor: '#fff',
                           borderColor: '#333', borderWidth: 1,
                           callbacks: { label: c => c.datasetIndex === 0 ? ' $'+c.parsed.y.toFixed(4) : '' } } }
        },
        plugins: [{ id: 'dynColor', afterLayout: c => {
            const { ctx: cx, chartArea, scales } = c;
            if (!chartArea) return;
            const g = getGradient(cx, chartArea, scales);
            c.data.datasets[0].borderColor = g;
        }}]
    });
}

function filterChart(hours, btn) {
    document.querySelectorAll('.btn-t').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeChartHours = hours;
    localStorage.setItem('edgebot_chart_window', hours);
    applyChartFilter();
}

function applyChartFilter() {
    if (activeChartHours === 0 || allTimestamps.length === 0) {
        const tradeIdx = allData.map((v,i) => i > 0 ? i : -1).filter(i => i >= 0);
        buildChart(allLabels, allData, tradeIdx);
        return;
    }
    const cutoff = Date.now() - activeChartHours * 3600000;
    const idx = allTimestamps.map((ts,i) => new Date(ts).getTime() >= cutoff ? i : -1).filter(i => i >= 0);
    const filteredData   = idx.map(i => allData[i]);
    const tradeIdx = filteredData.map((v,i) => i > 0 ? i : -1).filter(i => i >= 0);
    buildChart(idx.map(i => allLabels[i]), filteredData, tradeIdx);
}

function switchWindow(w, btn) {
    document.querySelectorAll('.t-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    localStorage.setItem('edgebot_stats_window', w);
    ['all','24h','7d'].forEach(id => {
        document.getElementById('stats-'+id).style.display = id === w ? '' : 'none';
    });
}

// ── Apply persisted state on load ─────────────────────────────────────────────
(function applyPersistedState() {
    // Chart time window
    const chartBtns = document.querySelectorAll('.btn-t');
    const hoursMap = {'0':0,'1':1,'6':6,'24':24};
    chartBtns.forEach(b => {
        const h = hoursMap[_savedChart];
        const label = b.textContent.trim();
        if ((_savedChart === '0' && label === 'ALL') ||
            (_savedChart === '1' && label === '1H') ||
            (_savedChart === '6' && label === '6H') ||
            (_savedChart === '24' && label === '24H')) {
            b.classList.add('active');
        } else {
            b.classList.remove('active');
        }
    });
    // Stats window
    ['all','24h','7d'].forEach(id => {
        document.getElementById('stats-'+id).style.display = id === _savedStats ? '' : 'none';
    });
    document.querySelectorAll('.t-btn').forEach(b => {
        b.classList.toggle('active', b.textContent.trim().toLowerCase() === _savedStats ||
                                     (b.textContent.trim() === 'All' && _savedStats === 'all'));
    });
    applyChartFilter();
})();

// ── Countdown timer ───────────────────────────────────────────────────────────
let countdownVal = 10;
let countdownInterval = null;

function startCountdown() {
    countdownVal = 10;
    const el = document.getElementById('countdown');
    if (el) el.textContent = countdownVal;
    countdownInterval = setInterval(() => {
        countdownVal--;
        const el = document.getElementById('countdown');
        if (el) el.textContent = Math.max(0, countdownVal);
    }, 1000);
}
startCountdown();

// ── Pause / resume auto-refresh ───────────────────────────────────────────────
let paused = false;
let pauseTimer = null;

function togglePause() {
    paused = !paused;
    const btn    = document.getElementById('pauseBtn');
    const status = document.getElementById('refresh-status');
    const meta   = document.getElementById('refresh-meta');
    if (paused) {
        btn.textContent = '▶ Resume';
        btn.classList.add('paused');
        status.innerHTML = '<span style="color:var(--yellow);">paused</span>';
        meta.removeAttribute('content');
        if (countdownInterval) { clearInterval(countdownInterval); countdownInterval = null; }
        pauseTimer = setTimeout(() => { if (paused) togglePause(); }, 300000);
    } else {
        btn.textContent = '⏸ Pause';
        btn.classList.remove('paused');
        status.innerHTML = '↻ <span id="countdown">10</span>s';
        meta.setAttribute('content', '10');
        if (pauseTimer) { clearTimeout(pauseTimer); pauseTimer = null; }
        startCountdown();
        location.reload();
    }
}
</script>
</body>
</html>
"""


def fetch_live_balance() -> tuple:
    """
    Fetch live Kalshi balance. Returns (balance_dollars: float, from_api: bool).

    The Kalshi GET /portfolio/balance endpoint returns:
      {"balance": <int cents>, "portfolio_value": <int cents>, "updated_ts": <int>}
    There is no "balance_dollars" field — balance is always integer cents.
    """
    if not _kalshi_ok or _kalshi is None:
        return 0.0, False
    try:
        loop = asyncio.new_event_loop()
        resp = loop.run_until_complete(_kalshi.get_balance())
        loop.close()
        if "balance" in resp:
            return resp["balance"] / 100.0, True
    except Exception as e:
        print(f"[DASHBOARD] Balance fetch failed: {e}")
    return 0.0, False


_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

def load_config() -> dict:
    try:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def compute_stats(pnl_series: pd.Series):
    """Compute profit factor and per-trade Sharpe from a series of PnL values."""
    if pnl_series.empty:
        return 0.0, 0.0, 0.0, 0.0, 0, 0

    wins_   = pnl_series[pnl_series > 0]
    losses_ = pnl_series[pnl_series < 0]

    gross_win  = wins_.sum()
    gross_loss = abs(losses_.sum())

    pf     = (gross_win / gross_loss) if gross_loss > 0 else 9999.0
    pf     = min(pf, 9999.0)
    mean_  = pnl_series.mean()
    std_   = pnl_series.std()
    sharpe = (mean_ / std_) if std_ > 0 else 0.0

    return pf, sharpe, gross_win, gross_loss, len(wins_), len(losses_)


def build_config_sections(cfg: dict) -> dict:
    # Note: TIME_ENTRY_MIN_MIN is the upper bound on minutes_left (entry window opens when < this)
    #       TIME_ENTRY_MAX_MIN is the lower bound on minutes_left (entry window closes when < this)
    sections = {
        "Trading": ["PAPER_MODE", "MAX_FILLS_PER_SESSION", "MAX_CONTRACTS_LIMIT",
                    "MAKER_MAX_ENTRY_PRICE", "ENTRY_TTL_SECONDS",
                    "TIME_ENTRY_MIN_MIN", "TIME_ENTRY_MAX_MIN", "STARTING_DEPOSIT"],
        "ML Signal": ["ML_CONFIDENCE_TAU", "ML_INFERENCE_WINDOW_MIN",
                      "ML_SPREAD_MAX_CENTS", "MAX_STRIKE_DISADVANTAGE_BPS"],
        "Kelly":   ["KELLY_FRACTION"],
        "Stop":    ["STOP_TRAIL_CENTS", "STOP_FLOOR_CENTS",
                    "STOP_TL_SH_THRESHOLD", "STOP_TL_LG_THRESHOLD",
                    "STOP_DELAY_SH", "STOP_DELAY_MD", "STOP_DELAY_LG",
                    "STOP_EXIT_MAX_RETRIES", "STOP_EXIT_RETRY_INCREMENT"],
        "Execution": ["REPRICE_THRESHOLD", "AMEND_COOLDOWN_SEC", "TAKER_FORCE_MIN_LEFT",
                      "LADDER_THRESHOLD", "LADDER_FILL_FRACTION"],
        "System":  ["MAX_DAILY_LOSS", "MAX_ORDERBOOK_STALE_SEC", "HEARTBEAT_INTERVAL_SEC",
                    "SETTLEMENT_INITIAL_DELAY", "SETTLEMENT_MAX_RETRIES",
                    "STOP_CONFIRM_DELAY_SEC"],
    }
    result = {}
    for section, keys in sections.items():
        items = [(k, cfg.get(k, "—")) for k in keys if k in cfg]
        if items:
            result[section] = items
    return result


_PROJECT_ROOT   = Path(__file__).resolve().parent.parent
_BTC_CSV        = _PROJECT_ROOT / "data" / "btc_1min.csv"
_ARTIFACTS_DIR  = _PROJECT_ROOT / "artifacts"


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


def _collector_status() -> dict:
    """Return live collector health from the last row of btc_1min.csv."""
    import subprocess as _sp
    try:
        result = _sp.run(["tail", "-n", "1", str(_BTC_CSV)],
                         capture_output=True, text=True, timeout=5)
        ts_str = result.stdout.strip().split(",")[0]
        last_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        stale   = (datetime.now(timezone.utc) - last_ts).total_seconds() > 180
        last_ct = last_ts.astimezone(pytz.timezone("US/Central")).strftime("%H:%M:%S")

        today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        gr = _sp.run(["grep", "-c", today_prefix, str(_BTC_CSV)],
                     capture_output=True, text=True, timeout=10)
        rows_today = int(gr.stdout.strip()) if gr.returncode in (0, 1) else 0

        return {"collector_last_ts": last_ct,
                "collector_rows_today": rows_today,
                "collector_stale": stale}
    except Exception:
        return {"collector_last_ts": "--", "collector_rows_today": 0, "collector_stale": True}


def _model_info() -> dict:
    """Return last training timestamp and metrics from artifacts/."""
    from datetime import timedelta
    central = pytz.timezone("US/Central")
    try:
        flag_path = _ARTIFACTS_DIR / "model_updated.flag"
        if flag_path.exists():
            trained_at_raw = flag_path.read_text().strip()
            trained_at = datetime.fromisoformat(trained_at_raw)
        else:
            model_path = _ARTIFACTS_DIR / "two_class_model.joblib"
            if model_path.exists():
                trained_at = datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc)
            else:
                trained_at = None

        trained_str = (trained_at.astimezone(central).strftime("%Y-%m-%d %H:%M CT")
                       if trained_at else "--")

        metrics = {}
        mp = _ARTIFACTS_DIR / "metrics_two_class.json"
        if mp.exists():
            with open(mp) as f:
                metrics = json.load(f)

        now_ct     = datetime.now(central)
        days_ahead = (6 - now_ct.weekday()) % 7 or 7
        next_sun   = (now_ct + timedelta(days=days_ahead)).replace(
                         hour=0, minute=0, second=0, microsecond=0)
        next_retrain = next_sun.strftime("%a %b %-d %H:%M CT")

        # Prefer the grid-search optimized threshold over the training-time default
        dt_path = _ARTIFACTS_DIR / "decision_threshold.json"
        live_tau = metrics.get("confidence_tau", "--")
        try:
            if dt_path.exists():
                with open(dt_path) as _f:
                    live_tau = float(json.load(_f)["confidence_tau"])
        except Exception:
            pass

        model_coverage = round(metrics.get("coverage", 0) * 100, 2)
        return {
            "model_trained_at":  trained_str,
            "model_dir_acc":     round(metrics.get("direction_accuracy", 0) * 100, 1),
            "model_win_rate":    round(metrics.get("win_rate", 0) * 100, 1),
            "model_coverage":    model_coverage,
            "model_atr_filtered": model_coverage == 0.0 and metrics.get("n_total", 0) > 0,
            "model_tau":         live_tau,
            "next_retrain":      next_retrain,
        }
    except Exception:
        return {"model_trained_at": "--", "model_dir_acc": 0.0, "model_win_rate": 0.0,
                "model_coverage": 0.0, "model_atr_filtered": False,
                "model_tau": "--", "next_retrain": "--"}


def get_data():
    files = glob.glob(f"{CSV_FILE}*")
    df_list = []
    for p in files:
        if not os.path.exists(p):
            continue
        try:
            tmp = pd.read_csv(p)
            if "timestamp" in tmp.columns and not tmp.empty:
                df_list.append(tmp)
        except Exception:
            pass

    if not df_list:
        return None

    try:
        df = pd.concat(df_list, ignore_index=True)
        if "timestamp" not in df.columns or df.empty:
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        last    = df.iloc[-1]
        central = pytz.timezone("US/Central")
        cfg     = load_config()

        starting_deposit = safe_float(cfg.get("STARTING_DEPOSIT", 1000.0))

        # ── Live ATR ───────────────────────────────────────────────────────────────
        try:
            _btc_df = pd.read_csv(_BTC_CSV).tail(100)
            live_atr = _calculate_live_atr_14(_btc_df)
        except Exception:
            live_atr = 0.0

        # ── Active Kelly risk ──────────────────────────────────────────────────────
        KELLY_WIN_RATE  = safe_float(cfg.get("KELLY_WIN_RATE", 0.55))
        KELLY_FRACTION  = safe_float(cfg.get("KELLY_FRACTION", 0.25))
        kelly_full      = (KELLY_WIN_RATE - (1.0 - KELLY_WIN_RATE)) / 1.0
        active_risk_pct = max(0.0, kelly_full) * KELLY_FRACTION * 100

        # ── Live balance: API first, then CSV fallback ─────────────────────────────
        live_balance, balance_from_api = fetch_live_balance()
        if not balance_from_api:
            sync_rows = df[df["event"] == "BALANCE_SYNC"]
            if not sync_rows.empty:
                # Use bankroll column from most recent BALANCE_SYNC row
                live_balance = safe_float(sync_rows.iloc[-1].get("bankroll", 0.0))
            else:
                live_balance = safe_float(last.get("bankroll", 0.0))

        # ── Closed trade rows — de-duplicated per trade ────────────────────────────
        # Each live trade generates: STOP_EXIT then STOP_CONFIRMED (or PAYOUT/SETTLE).
        # Count only STOP_CONFIRMED for live stops to avoid double-counting.
        # Paper mode generates only STOP_EXIT (no STOP_CONFIRMED), so include those.
        paper_stop_exits = df[
            (df["event"] == "STOP_EXIT") & (df["mode"] == "PAPER")
        ].copy()

        settled = df[df["event"].isin(["PAYOUT", "SETTLE", "STOP_CONFIRMED", "STOP_FAILED_EXPIRY"])].copy()
        settled = pd.concat([settled, paper_stop_exits], ignore_index=True)
        settled = settled.sort_values("timestamp").reset_index(drop=True)

        if "pnl_this_trade" in settled.columns and not settled.empty:
            settled["pnl_float"] = settled["pnl_this_trade"].apply(safe_float)
            settled["cum_pnl"]   = settled["pnl_float"].cumsum()
        else:
            settled["pnl_float"] = 0.0
            settled["cum_pnl"]   = 0.0

        # ── Concept drift: rolling 50-trade win rate ───────────────────────────────
        # Must run after pnl_float is added above.
        recent_50 = settled.tail(50)
        if len(recent_50) > 0 and "pnl_float" in settled.columns:
            drift_win_rate = (recent_50["pnl_float"] > 0).sum() / len(recent_50) * 100
        else:
            drift_win_rate = 0.0

        # ── Reconciled PnL — prefer Kalshi-verified amounts over estimates ─────────
        # RECONCILE rows carry the actual PnL pulled from Kalshi after settlement.
        # For trades that have a RECONCILE row, use reconcile_actual_pnl.
        # For trades not yet reconciled, fall back to pnl_this_trade from settled.
        reconcile_rows = df[df["event"] == "RECONCILE"].copy() if "RECONCILE" in df["event"].values else pd.DataFrame()
        has_reconcile_col = "reconcile_actual_pnl" in df.columns

        if not reconcile_rows.empty and has_reconcile_col:
            reconcile_rows["rec_pnl_float"] = reconcile_rows["reconcile_actual_pnl"].apply(safe_float)
            reconciled_tickers = set(reconcile_rows["ticker"].dropna().unique())
            rec_pnl_sum   = reconcile_rows["rec_pnl_float"].sum()
            unreconciled  = settled[~settled["ticker"].isin(reconciled_tickers)]
            realized_pnl  = rec_pnl_sum + unreconciled["pnl_float"].sum()
            reconcile_count = len(reconcile_rows)
            # Build a cumulative chart series using reconcile_actual_pnl where available
            # Merge reconcile amounts back onto settled by ticker for chart continuity
            rec_lookup = reconcile_rows.groupby("ticker")["rec_pnl_float"].last()
            settled["chart_pnl"] = settled.apply(
                lambda r: rec_lookup.get(r["ticker"], r["pnl_float"]) if r["ticker"] in reconciled_tickers else r["pnl_float"],
                axis=1
            )
            settled["cum_pnl"] = settled["chart_pnl"].cumsum()
        else:
            realized_pnl    = settled["pnl_float"].sum()
            reconcile_count = 0
            settled["chart_pnl"] = settled["pnl_float"]

        # divergence flag: any RECONCILE row where |delta| >= threshold
        reconcile_divergence = False
        reconcile_divergence_count = 0
        if not reconcile_rows.empty and "reconcile_delta" in reconcile_rows.columns:
            threshold = safe_float(cfg.get("RECONCILE_DIVERGENCE_THRESHOLD", 0.10))
            divs = reconcile_rows[reconcile_rows["reconcile_delta"].apply(safe_float).abs() >= threshold]
            reconcile_divergence       = len(divs) > 0
            reconcile_divergence_count = len(divs)

        gross_proceeds_total = settled["gross_proceeds"].apply(safe_float).sum() if "gross_proceeds" in settled.columns else 0.0
        gross_cost_total     = settled["gross_cost"].apply(safe_float).sum()     if "gross_cost"     in settled.columns else 0.0
        total_fees           = settled["net_fees"].apply(safe_float).sum()       if "net_fees"       in settled.columns else 0.0

        roi = ((live_balance - starting_deposit) / starting_deposit * 100) if starting_deposit > 0 else 0.0

        # ── PnL chart — anchor start to bot's first log entry ─────────────────────
        first_ts         = df["timestamp"].iloc[0]
        _time_span   = (settled["timestamp"].max() - settled["timestamp"].min()).total_seconds() if not settled.empty else 0
        _label_fmt   = "%m/%d %H:%M" if _time_span > 86400 else "%H:%M"
        chart_labels = settled["timestamp"].dt.tz_convert(central).dt.strftime(_label_fmt).tolist()
        chart_data       = settled["cum_pnl"].round(6).tolist()
        chart_timestamps = settled["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
        chart_labels     = [first_ts.astimezone(central).strftime("%H:%M")] + chart_labels
        chart_data       = [0.0] + chart_data
        chart_timestamps = [first_ts.strftime("%Y-%m-%dT%H:%M:%SZ")] + chart_timestamps

        # ── Win / loss — one count per closed trade ────────────────────────────────
        wins   = len(df[df["event"] == "PAYOUT"])
        losses = (
            len(df[df["event"] == "SETTLE"]) +
            len(df[df["event"] == "STOP_CONFIRMED"]) +
            len(df[df["event"] == "STOP_FAILED_EXPIRY"]) +
            len(df[(df["event"] == "STOP_EXIT") & (df["mode"] == "PAPER")])
        )
        total_trades = wins + losses
        win_rate     = (wins / total_trades * 100) if total_trades > 0 else 0.0

        # ── Rolling windows ────────────────────────────────────────────────────────
        now_utc    = pd.Timestamp.now("UTC")
        cutoff_24h = now_utc - pd.Timedelta(hours=24)
        cutoff_7d  = now_utc - pd.Timedelta(days=7)

        recent_24h       = settled[settled["timestamp"] >= cutoff_24h]
        recent_7d        = settled[settled["timestamp"] >= cutoff_7d]
        rolling_loss_24h = abs(recent_24h["pnl_float"][recent_24h["pnl_float"] < 0].sum())

        # BUGS 6+7 FIX: compute_stats was using raw pnl_float from settled rows, which for
        # STOP_CONFIRMED rows contained wrong values (Bug 1 - incorrect NO position PnL).
        # Use chart_pnl instead, which has already been reconcile-corrected above for any
        # trade that has a RECONCILE row. This ensures gross_win, gross_loss and profit
        # factor reflect Kalshi-verified values rather than the bot's raw computation.
        pf_all, sharpe_all, gw_all, gl_all, w_all, l_all = compute_stats(settled["chart_pnl"])
        pf_24h, sharpe_24h, gw_24h, gl_24h, w_24h, l_24h = compute_stats(
            recent_24h["chart_pnl"] if not recent_24h.empty else pd.Series(dtype=float))
        pf_7d,  sharpe_7d,  gw_7d,  gl_7d,  w_7d,  l_7d  = compute_stats(
            recent_7d["chart_pnl"]  if not recent_7d.empty  else pd.Series(dtype=float))

        # ── Trade quality ──────────────────────────────────────────────────────────
        fill_rows  = df[df["event"] == "FILL_CONFIRMED"].copy() if "FILL_CONFIRMED" in df["event"].values else pd.DataFrame()
        order_rows = df[df["event"].isin(["ORDER_RESTING", "LIVE_BUY", "PAPER_BUY"])].copy()

        # ── avg_entry: use FILL_CONFIRMED rows with a real fill qty — not all
        #    ORDER_RESTING rows (which include cancelled/repriced attempts).
        #    Fall back to order_rows if no fills recorded yet (e.g. paper mode).
        actual_fills = (
            fill_rows[fill_rows["kalshi_fill_qty"].apply(safe_float) > 0].copy()
            if not fill_rows.empty and "kalshi_fill_qty" in fill_rows.columns
            else pd.DataFrame()
        )
        if not actual_fills.empty:
            avg_entry      = round(actual_fills["entry_price"].apply(safe_float).mean(), 1)
            avg_signal_age = round(actual_fills["signal_age_min"].apply(safe_float).mean(), 1) if "signal_age_min" in actual_fills.columns else 0
        elif len(order_rows):
            avg_entry      = round(order_rows["entry_price"].apply(safe_float).mean(), 1)
            avg_signal_age = round(order_rows["signal_age_min"].apply(safe_float).mean(), 1) if "signal_age_min" in order_rows.columns else 0
        else:
            avg_entry      = 0
            avg_signal_age = 0

        # avg ML confidence across actual fills
        avg_ml_confidence = 0.0
        if not actual_fills.empty and "ml_confidence" in actual_fills.columns:
            avg_ml_confidence = round(actual_fills["ml_confidence"].apply(safe_float).mean() * 100, 1)
        elif not order_rows.empty and "ml_confidence" in order_rows.columns:
            avg_ml_confidence = round(order_rows["ml_confidence"].apply(safe_float).mean() * 100, 1)

        avg_pnl        = round(settled["pnl_float"].mean(), 6) if not settled.empty else 0.0
        avg_fees       = round(settled["net_fees"].apply(safe_float).mean(), 6) if "net_fees" in settled.columns and not settled.empty else 0.0

        # ── avg_slippage: compare first posted ORDER_RESTING price vs actual fill price.
        #    BUG-8 FIX: Previously computed kalshi_fill_price - entry_price within the same
        #    FILL_CONFIRMED row, but both columns were set to actual_entry (same variable),
        #    so slippage was always 0. Now we join the earliest ORDER_RESTING price for each
        #    fill's order_id to measure true repricing drift (how much the bot chased).
        avg_slippage = 0.0
        if not actual_fills.empty and "kalshi_fill_price" in actual_fills.columns and "order_id" in actual_fills.columns:
            if not order_rows.empty and "order_id" in order_rows.columns:
                # For each order_id, find the FIRST (earliest) resting price posted
                first_post = (
                    order_rows.sort_values("timestamp")
                    .drop_duplicates(subset=["order_id"], keep="first")
                    [["order_id", "entry_price"]]
                    .rename(columns={"entry_price": "posted_price"})
                )
                slip_df = (
                    actual_fills
                    .drop_duplicates(subset=["order_id"], keep="first")
                    .merge(first_post, on="order_id", how="left")
                )
                slip_df["slippage"] = (
                    slip_df["kalshi_fill_price"].apply(safe_float) -
                    slip_df["posted_price"].apply(safe_float)
                )
                valid_slip = slip_df["slippage"].dropna()
                avg_slippage = round(valid_slip.mean(), 1) if not valid_slip.empty else 0.0
            else:
                # Fallback: no ORDER_RESTING rows available (e.g. paper mode)
                avg_slippage = 0.0

        # ── Bot health ─────────────────────────────────────────────────────────────
        is_active    = (now_utc - last["timestamp"]).total_seconds() < 120
        # Use only the single most recent row for ob_stale — not a majority vote
        ob_stale     = bool(safe_int(last.get("ob_stale", 0))) if "ob_stale" in df.columns else False
        ob_delta_age = safe_float(last.get("ob_last_delta_age", 9999))
        ws_connected = bool(safe_int(last.get("ws_connected", 0))) if "ws_connected" in df.columns else True
        mode_str     = str(last.get("mode", "PAPER"))

        # ML signal vars — from ml_* columns (v8 log format)
        ml_direction_raw = last.get("ml_direction", None)
        try:
            ml_direction = int(ml_direction_raw) if ml_direction_raw not in (None, "", "None") else None
        except (ValueError, TypeError):
            ml_direction = None
        ml_confidence = safe_float(last.get("ml_confidence", 0.0))
        ml_proba_up   = safe_float(last.get("ml_proba_up", 0.0))
        ml_birth_ts_raw = safe_float(last.get("ml_birth_ts", 0))
        # Guard against column misalignment: ml_birth_ts must be a plausible Unix timestamp
        # (between 2001 and 2040). Signal age minutes would be tiny (<1000) or zero.
        ml_birth_ts = ml_birth_ts_raw if 1_000_000_000 < ml_birth_ts_raw < 2_000_000_000 else 0.0
        ml_signal_age = round((time.time() - ml_birth_ts) / 60.0, 1) if ml_birth_ts > 0 else None
        ml_fired_at   = (
            pd.Timestamp(ml_birth_ts, unit="s", tz="UTC")
            .astimezone(pytz.timezone("US/Central"))
            .strftime("%H:%M:%S") if ml_birth_ts > 0 else "--"
        )

        # Runtime tau: what the bot is actually using right now (config.json is authoritative)
        _dt_path = _ARTIFACTS_DIR / "decision_threshold.json"
        ml_tau = safe_float(cfg.get("ML_CONFIDENCE_TAU", 0.50))
        # Trained tau: what the last grid-search produced (shown separately in Model card)
        trained_tau = ml_tau
        try:
            if _dt_path.exists():
                with open(_dt_path) as _dtf:
                    trained_tau = float(json.load(_dtf)["confidence_tau"])
        except Exception:
            pass

        atr_min = safe_float(cfg.get("ATR_MIN", 15.0))
        atr_max = safe_float(cfg.get("ATR_MAX", 30.0))

        birth_ts   = ml_birth_ts
        signal_age = ml_signal_age  # None when no signal has fired

        _ticker_rows = df[df["ticker"].notna() & (df["ticker"] != "")] if not df.empty else pd.DataFrame()
        lm        = _ticker_rows.iloc[-1] if not _ticker_rows.empty else last
        yes_bid   = safe_int(lm.get("raw_yes_bid", 0))
        no_bid    = safe_int(lm.get("raw_no_bid",  0))
        yes_ask   = safe_int(lm.get("ask_yes", 100 - yes_bid if yes_bid else 99))
        no_ask    = safe_int(lm.get("ask_no",  100 - no_bid  if no_bid  else 99))

        # Moneyness bps relative to current direction
        # For YES: (strike - spot) / spot * 10000  (positive = strike above spot = bad for YES)
        # For NO:  (spot - strike) / spot * 10000  (positive = strike below spot = bad for NO)
        _spot   = safe_float(last.get("btc_price", 0))
        _strike = safe_float(lm.get("strike", 0))
        if _spot > 0 and _strike > 0 and ml_direction is not None:
            if ml_direction == 1:
                moneyness_bps = (_strike - _spot) / _spot * 10_000.0
            else:
                moneyness_bps = (_spot - _strike) / _spot * 10_000.0
        else:
            moneyness_bps = 0.0
        yes_liq   = safe_int(lm.get("yes_liq", 0))
        no_liq    = safe_int(lm.get("no_liq",  0))
        btc_price = safe_float(last.get("btc_price", 0))

        # ── Active position/order detection ────────────────────────────────────────
        # Close events: STOP_CONFIRMED (not STOP_EXIT) to avoid flicker.
        # Paper mode: STOP_EXIT is the close event since no STOP_CONFIRMED is written.
        closed_events   = ["PAYOUT", "SETTLE", "STOP_CONFIRMED", "STOP_FAILED_EXPIRY", "ORDER_UNFILLED"]
        last_closed_ts  = df[df["event"].isin(closed_events)]["timestamp"].max() if not df.empty else pd.NaT
        paper_stop_ts   = df[(df["event"] == "STOP_EXIT") & (df["mode"] == "PAPER")]["timestamp"].max() if not df.empty else pd.NaT
        if pd.notna(paper_stop_ts) and (pd.isna(last_closed_ts) or paper_stop_ts > last_closed_ts):
            last_closed_ts = paper_stop_ts

        last_opened_ts  = df[df["event"].isin(["FILL_CONFIRMED", "PAPER_BUY"])]["timestamp"].max() if not df.empty else pd.NaT
        last_resting_ts = df[df["event"] == "ORDER_RESTING"]["timestamp"].max() if not df.empty else pd.NaT

        has_active_position = pd.notna(last_opened_ts) and (pd.isna(last_closed_ts) or last_opened_ts > last_closed_ts)
        has_active_order    = (not has_active_position) and pd.notna(last_resting_ts) and (pd.isna(last_closed_ts) or last_resting_ts > last_closed_ts)

        stop_trail = stop_best_bid = stop_active = stop_level = 0
        if "stop_trail" in df.columns:
            _stop_src = df[df["stop_trail"].apply(safe_int) > 0]
            if not _stop_src.empty:
                lh            = _stop_src.iloc[-1]
                stop_trail    = safe_int(lh.get("stop_trail", 0))
                stop_best_bid = safe_int(lh.get("stop_best_bid", 0))
                stop_active   = bool(safe_int(lh.get("stop_active", 0)))
                stop_level    = max(stop_best_bid - stop_trail, safe_int(cfg.get("STOP_FLOOR_CENTS", 15)))

        pos_ticker = pos_side = pos_entry = pos_qty = pos_fill_price = pos_fill_qty = pos_cost = pos_fees = ""
        if has_active_position and not fill_rows.empty:
            lf             = fill_rows.iloc[-1]
            pos_ticker     = str(lf.get("ticker", ""))
            pos_side       = str(lf.get("side", ""))
            pos_entry      = str(safe_int(lf.get("entry_price", 0)))
            pos_qty        = str(safe_int(lf.get("qty", 0)))
            pos_fill_price = str(safe_int(lf.get("kalshi_fill_price", 0)))
            pos_fill_qty   = str(round(safe_float(lf.get("kalshi_fill_qty", 0))))
            pos_cost       = f"{safe_float(lf.get('kalshi_fill_cost', 0)):.4f}"
            pos_fees       = f"{safe_float(lf.get('kalshi_fees', 0)):.6f}"

        order_ticker = order_side = order_price = order_qty = order_age = order_amends = order_coid = ""
        if has_active_order:
            ro           = df[df["event"] == "ORDER_RESTING"].iloc[-1]
            order_ticker = str(ro.get("ticker", ""))
            order_side   = str(ro.get("side", ""))
            order_price  = str(safe_int(ro.get("entry_price", 0)))
            order_qty    = str(safe_int(ro.get("qty", 0)))
            order_age    = str(round((now_utc - ro["timestamp"]).total_seconds()))
            order_coid   = str(ro.get("client_order_id", ""))
            amend_rows   = df[(df["event"] == "ORDER_AMENDED") & (df["timestamp"] >= ro["timestamp"])]
            order_amends = str(len(amend_rows))

        show_events = ["ML_INFERENCE", "PAPER_BUY", "LIVE_BUY", "ORDER_RESTING", "ORDER_AMENDED",
                       "ORDER_ESCALATED", "TAKER_ESCALATION", "LADDER_PARTIAL",
                       "ORDER_UNFILLED", "FILL_CONFIRMED", "FILL_VERIFIED",
                       "PAYOUT", "SETTLE", "STOP_EXIT", "STOP_CONFIRMED", "STOP_FAILED",
                       "STOP_FAILED_EXPIRY", "STOP_EXPIRY_RISK", "STOP_PARTIAL", "SETTLE_VERIFIED",
                       "RECONCILE", "CIRCUIT_BREAKER", "MODEL_RELOAD", "ERROR"]
        log_df = df[df["event"].isin(show_events)].tail(50).iloc[::-1]
        logs = []
        for _, r in log_df.iterrows():
            ev  = str(r.get("event", ""))
            msg = str(r.get("msg", ""))
            # SETTLE_VERIFIED rows omit stop_active, shifting columns left by one;
            # the settlement text lands in reconcile_delta instead of msg.
            if ev == "SETTLE_VERIFIED" and msg in ("nan", "", "None"):
                msg = str(r.get("reconcile_delta", ""))

            # For SETTLE_VERIFIED, derive a clean label and color from the msg content
            # so sessions with no position still show a clear YES/NO outcome.
            if ev == "SETTLE_VERIFIED":
                if "Market settled YES" in msg:
                    display_event = "SETTLED YES"
                    color         = "#00e676"   # green — would have won
                elif "Market settled NO" in msg:
                    display_event = "SETTLED NO"
                    color         = "#ff5252"   # red — would have lost
                else:
                    # Old-format log row (pre-fix) — show neutrally
                    display_event = "SETTLED"
                    color         = "#448aff"
            else:
                display_event = ev.replace("PAPER_", "").replace("LIVE_", "")
                color         = _event_color(ev)

            logs.append({
                "time":  r["timestamp"].astimezone(central).strftime("%H:%M:%S"),
                "event": display_event,
                "color": color,
                "msg":   msg,
            })

        # ── Latest ML inferences ───────────────────────────────────────────────
        infer_rows   = df[df["event"] == "ML_INFERENCE"].tail(10)
        fill_events  = df[df["event"].isin(["FILL_CONFIRMED", "PAPER_BUY"])]

        settle_events = df[df["event"] == "SETTLE_VERIFIED"]
        settle_by_ticker = {}
        for _, sv in settle_events.iterrows():
            sv_ticker = str(sv.get("ticker", ""))
            # SETTLE_VERIFIED rows omit stop_active, shifting columns left by one;
            # the settlement text ends up in reconcile_delta instead of msg.
            sv_text = " ".join(str(sv.get(col, "")) for col in ("msg", "reconcile_delta"))
            if "Market settled YES" in sv_text:
                settle_by_ticker[sv_ticker] = "YES"
            elif "Market settled NO" in sv_text:
                settle_by_ticker[sv_ticker] = "NO"

        inference_history = []
        for _, ir in infer_rows.iloc[::-1].iterrows():
            iconf    = safe_float(ir.get("ml_confidence", 0.0))
            idir_raw = ir.get("ml_direction", None)
            try:
                idir = int(idir_raw) if idir_raw not in (None, "", "None") else None
            except (ValueError, TypeError):
                idir = None
            dir_str = "UP" if idir == 1 else "DN" if idir == 0 else "--"

            # Use the tau that was active at inference time (logged as ml_tau column).
            # Guard: valid tau must be in (0, 1]. Values outside that range mean the column
            # was misaligned (e.g. a Unix timestamp or signal_age_min landed there).
            row_tau = safe_float(ir.get("ml_tau", 0.0))
            effective_tau = row_tau if 0 < row_tau <= 1.0 else ml_tau
            if iconf < effective_tau:
                status = "tau_gate_fail"
            else:
                window_end      = ir["timestamp"] + pd.Timedelta(minutes=6)
                following_fills = fill_events[
                    (fill_events["timestamp"] > ir["timestamp"]) &
                    (fill_events["timestamp"] <= window_end)
                ]
                status = "success" if not following_fills.empty else "filtered"

            ir_ticker  = str(ir.get("ticker", ""))
            settlement = settle_by_ticker.get(ir_ticker)
            if idir is None:
                outcome = "--"
            elif settlement is None:
                outcome = "pending"
            elif (idir == 1 and settlement == "YES") or (idir == 0 and settlement == "NO"):
                outcome = "correct"
            else:
                outcome = "wrong"

            inference_history.append({
                "time":          ir["timestamp"].astimezone(central).strftime("%H:%M:%S"),
                "direction":     dir_str,
                "confidence":    round(iconf * 100, 1),
                "effective_tau": round(effective_tau * 100, 1),
                "status":        status,
                "outcome":       outcome,
            })

        config_sections = build_config_sections(cfg)
        last_ct         = last["timestamp"].astimezone(central)

        return dict(
            last_update=last_ct.strftime("%H:%M:%S"),
            is_active=is_active, ob_stale=ob_stale, ob_delta_age=round(ob_delta_age),
            ws_connected=ws_connected, mode=mode_str,
            live_balance=live_balance, balance_from_api=balance_from_api,
            starting_deposit=starting_deposit, roi=round(roi, 2),
            realized_pnl=realized_pnl,
            reconcile_count=reconcile_count,
            reconcile_divergence=reconcile_divergence,
            reconcile_divergence_count=reconcile_divergence_count,
            gross_proceeds_total=gross_proceeds_total,
            gross_cost_total=gross_cost_total, total_fees=total_fees,
            rolling_loss_24h=round(rolling_loss_24h, 4),
            ml_direction=ml_direction,
            ml_confidence=ml_confidence,
            ml_proba_up=ml_proba_up,
            ml_signal_age=ml_signal_age,
            ml_fired_at=ml_fired_at,
            ml_tau=ml_tau,
            trained_tau=trained_tau,
            atr_min=atr_min,
            atr_max=atr_max,
            live_atr=round(live_atr, 2),
            drift_win_rate=round(drift_win_rate, 1),
            active_risk_pct=round(active_risk_pct, 2),
            moneyness_bps=round(moneyness_bps, 1),
            signal_age=signal_age,
            ticker=str(lm.get("ticker", "--")),
            btc_price=btc_price,
            time_left=safe_float(lm.get("time_left", 0)),
            strike=safe_float(lm.get("strike", 0)),
            obi=safe_float(lm.get("obi", 0)),
            yes_bid=yes_bid, no_bid=no_bid, yes_ask=yes_ask, no_ask=no_ask,
            yes_liq=yes_liq, no_liq=no_liq,
            wins=wins, losses=losses, total_trades=total_trades, win_rate=win_rate,
            pf_all=pf_all,   sharpe_all=round(sharpe_all, 2),
            gross_win_all=gw_all,   gross_loss_all=gl_all,   wins_all=w_all,   losses_all=l_all,
            pf_24h=pf_24h,   sharpe_24h=round(sharpe_24h, 2),
            gross_win_24h=gw_24h,   gross_loss_24h=gl_24h,   wins_24h=w_24h,   losses_24h=l_24h,
            pf_7d=pf_7d,     sharpe_7d=round(sharpe_7d, 2),
            gross_win_7d=gw_7d,     gross_loss_7d=gl_7d,     wins_7d=w_7d,     losses_7d=l_7d,
            avg_entry=avg_entry, avg_slippage=avg_slippage,
            avg_signal_age=avg_signal_age, avg_ml_confidence=avg_ml_confidence,
            avg_pnl=avg_pnl, avg_fees=avg_fees,
            has_active_position=has_active_position, has_active_order=has_active_order,
            pos_ticker=pos_ticker, pos_side=pos_side, pos_entry=pos_entry, pos_qty=pos_qty,
            pos_fill_price=pos_fill_price, pos_fill_qty=pos_fill_qty,
            pos_cost=pos_cost, pos_fees=pos_fees,
            stop_trail=stop_trail, stop_best_bid=stop_best_bid,
            stop_level=stop_level, stop_active=stop_active,
            order_ticker=order_ticker, order_side=order_side, order_price=order_price,
            order_qty=order_qty, order_age=order_age, order_amends=order_amends,
            order_coid=order_coid or "—",
            chart_labels=chart_labels, chart_data=chart_data, chart_timestamps=chart_timestamps,
            logs=logs, config_sections=config_sections,
            inference_history=inference_history,
            **_collector_status(),
            **_model_info(),
        )

    except Exception as e:
        print(f"Dashboard error: {e}")
        import traceback; traceback.print_exc()
        return None


@app.route("/")
def home():
    data = get_data()
    return render_template_string(HTML_TEMPLATE, **data) if data else "Waiting for bot data..."

@app.route("/health")
def health():
    data = get_data()
    if not data:
        return jsonify({"status": "starting"}), 503
    return jsonify({
        "status":                    "ok",
        "last_update":               data["last_update"],
        "live_balance":              data["live_balance"],
        "balance_from_api":          data["balance_from_api"],
        "realized_pnl":              data["realized_pnl"],
        "reconcile_count":           data["reconcile_count"],
        "reconcile_divergence":      data["reconcile_divergence"],
        "reconcile_divergence_count":data["reconcile_divergence_count"],
        "roi":                       data["roi"],
        "win_rate":                  data["win_rate"],
        "ws_connected":              data["ws_connected"],
        "ob_delta_age":              data["ob_delta_age"],
    })

@app.route("/api/data")
def api_data():
    data = get_data()
    if not data:
        return jsonify({"error": "no data"}), 503
    light = {k: v for k, v in data.items()
             if k not in ("chart_labels", "chart_data", "chart_timestamps",
                          "logs", "config_sections")}
    return jsonify(light)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)
