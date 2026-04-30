# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Kalshi 15-minute BTC binary-options trading bot driven by a scikit-learn MLP trained on 1-minute Coinbase order-book data. The model predicts whether BTC price will be up or down at the end of each 15-minute Kalshi contract window.

## Environment setup

Always activate the venv before running any Python commands:
```bash
source venv/bin/activate
```

All commands run from the project root (`ML Bot/`). Credentials live in `.env` (never commit) and `kalshi.key` (never commit).

## Key commands

**Run the full system (all components supervised):**
```bash
python3 orchestrator.py
```

**Run components individually:**
```bash
python3 collector/btc_ws_collector.py       # continuous data collection
python3 bot/production_bot.py               # trading bot
python3 bot/dashboard.py                    # Flask monitoring UI
python3 bot/drift_monitor.py               # drift detection + auto-retrain
```

**ML pipeline (run in order):**
```bash
python3 ml/preprocess.py
python3 ml/train.py --optimize_threshold_for_profit --optimize_tau_by ev --atr_min 15.0 --atr_max 30.0
```

**ATR channel grid search (no artifacts written):**
```bash
for min in 5 10 15 20 25; do
    for max in 10 15 20 25 30 35 40; do
        if [ "$min" -lt "$max" ]; then
            python ml/train.py --optimize_threshold_for_profit --optimize_tau_by ev \
                --atr_min $min --atr_max $max 2>&1 | awk '/ATR volatility/{print} /TEST METRICS:/{flag=1} flag && /Coverage:/{print} flag && /Win Rate:/{print; flag=0}'
        fi
    done
done
```

Add `--epochs 5 --max_train_size 200000` for a faster sweep.

## Architecture

```
collector/btc_ws_collector.py   →   data/btc_1min.csv
ml/preprocess.py                →   data/processed/btc_1min_processed.parquet
ml/train.py                     →   artifacts/  (models, scaler, imputer, threshold)
bot/production_bot.py           →   reads artifacts/, trades on Kalshi
bot/dashboard.py                →   Flask monitoring UI
bot/drift_monitor.py            →   watches logs/production_log.csv for win-rate drift
orchestrator.py                 →   supervises all above as persistent subprocesses
```

### Data flow

1. **Collector** connects to Coinbase WebSocket, writes one row/minute to `data/btc_1min.csv` with OHLCV, buy/sell volume, CVD, and dual order-book snapshots (t=0s and t=30s).
2. **Preprocess** engineers features (volatility windows, microstructure signals, etc.) and writes `data/processed/btc_1min_processed.parquet`.
3. **Train** loads the parquet, selects top-128 features via mutual information, trains a sklearn `MLPClassifier`, optionally calibrates probabilities, and writes `.joblib` artifacts + `decision_threshold.json` to `artifacts/`.
4. **Bot** loads artifacts at startup. Each Kalshi session: runs ML inference once within the first `ML_INFERENCE_WINDOW_MIN` minutes, applies ATR gate, confidence gate, strike moneyness gate, then places a maker bid. Primary exit is hold-to-expiry; stop engine is the safety net.

### Hot-reload

After retraining, `train.py` writes `artifacts/model_updated.flag`. The bot's `model_watchdog` coroutine detects this flag and hot-reloads all artifacts without a restart. The same mechanism applies to `bot/config.json` — the bot polls it and updates `Config` fields live (except those in `_CONFIG_IMMUTABLE`).

### Drift detection

`drift_monitor.py` polls `logs/production_log.csv` every 5 minutes. If the rolling win rate over the last 50 PAYOUT/SETTLE events drops below 0.53, it triggers an emergency retrain (subject to a 12-hour cooldown).

### Orchestrator

`orchestrator.py` manages all four subprocesses with `ManagedProcess` (auto-restart on exit). It also schedules weekly retraining every Sunday at midnight CT via APScheduler.

## Critical files

| Path | Purpose |
|---|---|
| `bot/config.json` | Runtime config — hot-reloaded by the bot. `PAPER_MODE: false` enables live trading. |
| `artifacts/decision_threshold.json` | Confidence tau from grid-search — hot-reloaded by bot. |
| `artifacts/two_class_model.joblib` | Trained classifier (required at bot startup). |
| `artifacts/feature_schema.json` | Exact feature list the model was trained on — bot uses this to align inference features. |
| `bot/state/acted_birth_ts.json` | Persists per-session acted state across bot restarts. |
| `logs/production_log.csv` | Append-only trade log (ENTRY, FILL, PAYOUT, SETTLE events). |

## bot/config.json key fields

| Key | Description |
|---|---|
| `PAPER_MODE` | `true` = simulate trades only, `false` = live trading |
| `ML_CONFIDENCE_TAU` | Bot-side confidence floor (overrides `decision_threshold.json` if set higher) |
| `KELLY_FRACTION` | Fraction of Kelly criterion for position sizing |
| `MAX_CONTRACTS_LIMIT` | Hard cap on contracts per session |
| `MAX_FILLS_PER_SESSION` | Usually `1` — one trade per 15-min window |
| `ATR_MIN` / `ATR_MAX` | Live ATR gate — must match train.py `--atr_min`/`--atr_max` |

## train.py key flags

| Flag | Description |
|---|---|
| `--optimize_threshold_for_profit` | Grid-search confidence tau for max EV (writes `decision_threshold.json`) |
| `--optimize_tau_by ev\|winrate` | Optimization objective |
| `--atr_min` / `--atr_max` | Volatility channel filter (USD) — should match `bot/config.json` ATR gates |
| `--top_k_features` | Feature count for SelectKBest mutual-info selection |
| `--calibrate_probabilities` | Wrap model in `CalibratedClassifierCV` (Platt scaling) |
| `--epochs` | Training epochs (default 30) |

## Secrets — never commit

- `.env` — Kalshi API key and private key path
- `kalshi.key` — RSA private key for Kalshi API auth
