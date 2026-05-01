# ML Trading Bot

Kalshi 15-minute BTC up/down trading bot driven by a machine learning model trained on 1-minute Coinbase order book data.

## Architecture

```
collector/btc_ws_collector.py   →   data/btc_1min.csv
ml/preprocess.py                →   data/processed/btc_1min_processed.parquet
ml/train.py                     →   artifacts/  (two_class_model.joblib, scaler.joblib, imputer.joblib)
bot/production_bot.py           →   reads artifacts/, trades on Kalshi
bot/dashboard.py                →   Flask UI for monitoring
```

---

## Setup

**1. Activate the virtual environment** (required every terminal session):
```bash
source venv/bin/activate
```

**2. Configure credentials** — edit `.env` in the project root:
```
KALSHI_API_KEY=your_api_key_here
KALSHI_PRIVATE_KEY_PATH=/Users/yourname/Desktop/ML Bot/kalshi.key
```

All commands below are run from the project root (`ML Bot/`).

---

## Running the Data Collector

Connects to Coinbase and appends one row per minute to `data/btc_1min.csv`. Run this continuously in its own terminal.

```bash
python3 collector/btc_ws_collector.py
```

The collector reconnects automatically on dropped connections. Stop it with `Ctrl+C` (it finishes the current bar before exiting).

---

## ML Pipeline

Run these two steps in order. Both read and write to the project-root `data/` and `artifacts/` directories automatically — no path arguments needed unless you want to override.

### Step 1 — Preprocess

Reads `data/btc_1min.csv`, engineers features, writes `data/processed/btc_1min_processed.parquet`.

```bash
python3 ml/preprocess.py
```

Override paths if needed:
```bash
python3 ml/preprocess.py --input data/btc_1min.csv --output data/processed/
```

### Step 2 — Train

Reads `data/processed/btc_1min_processed.parquet`, trains the model, writes `.joblib` artifacts to `artifacts/`.

```bash
python3 ml/train.py --optimize_threshold_for_profit --optimize_tau_by ev --atr_min 15.0 --atr_max 30.0
```

`--optimize_threshold_for_profit` triggers a grid-search over candidate confidence thresholds and selects the one that maximises expected value (EV) on the hold-out set. The result is written to `artifacts/decision_threshold.json` and hot-reloaded by the bot's `model_watchdog` without a restart.

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--optimize_threshold_for_profit` | off | Enable grid-search tau optimisation (replaces `--confidence_tau`) |
| `--optimize_tau_by` | `ev` | Optimisation objective — `ev` (expected value) or `winrate` |
| `--atr_min` | `15.0` | Minimum ATR_14 (USD) for the volatility gate |
| `--atr_max` | `30.0` | Maximum ATR_14 (USD) for the volatility gate |
| `--horizon_min` | `15` | Prediction horizon in minutes |
| `--top_k_features` | `128` | Number of features selected for training |
| `--epochs` | `30` | Training epochs |

Training writes `two_class_model.joblib`, `scaler.joblib`, `imputer.joblib`, and `decision_threshold.json` to `artifacts/`. The bot will not start without the first three files.

### ATR channel sweep

Grid-searches all `(atr_min, atr_max)` combinations to find the best volatility channel. Prints the ATR gate stats, coverage, and win rate for each combination without saving artifacts.

```bash
for min in 5 10 15 20 25; do
    for max in 10 15 20 25 30 35 40; do
        if [ "$min" -lt "$max" ]; then
            echo "========================================="
            echo "TESTING CHANNEL: MIN $min | MAX $max"
            echo "========================================="
            python ml/train.py --optimize_threshold_for_profit --optimize_tau_by ev \
                --atr_min $min --atr_max $max 2>&1 | awk '
                /ATR volatility/ {print $0}
                /TEST METRICS:/ {flag=1}
                flag && /Coverage:/ {print $0}
                flag && /Win Rate:/ {print $0; flag=0}
            '
        fi
    done
done
```

Add `--epochs 5 --max_train_size 200000` for a faster sweep.

### Weekly retraining (every Sunday)

```bash
python3 ml/preprocess.py && python3 ml/train.py --optimize_threshold_for_profit --optimize_tau_by ev --atr_min 15.0 --atr_max 30.0
```

---

## Running the Bot

Requires `artifacts/` to be populated from a completed training run.

```bash
python3 bot/production_bot.py
```

The bot is in **paper mode by default** (`PAPER_MODE: true` in `bot/config.json`). Set it to `false` to trade live.

### Dashboard (optional, separate terminal)

```bash
python3 bot/dashboard.py
```

Opens a Flask UI for monitoring trades, P&L, and signal activity.

---

## Key Config — `bot/config.json`

| Key | Description |
|---|---|
| `PAPER_MODE` | `true` = simulate trades, `false` = live trading |
| `ML_CONFIDENCE_TAU` | Bot-side confidence threshold (gate on top of model output) |
| `KELLY_FRACTION` | Fraction of Kelly criterion used for position sizing |
| `MAX_CONTRACTS_LIMIT` | Hard cap on contracts per session |
| `MAKER_MAX_ENTRY_PRICE` | Maximum cents willing to pay as maker |

---

## File Reference

| Path | Purpose |
|---|---|
| `data/btc_1min.csv` | Raw 1-min OHLCV + order book data (collector writes here) |
| `data/processed/btc_1min_processed.parquet` | Engineered features (preprocess writes here) |
| `artifacts/two_class_model.joblib` | Trained classifier |
| `artifacts/scaler.joblib` | Feature scaler |
| `artifacts/imputer.joblib` | Missing value imputer |
| `artifacts/decision_threshold.json` | Grid-search optimised confidence tau (written by `train.py`, hot-reloaded by bot) |
| `logs/production_log.csv` | Trade log (bot writes here) |
| `logs/bot.log` | Bot system log |
| `.env` | Kalshi API credentials (never commit) |
| `kalshi.key` | Kalshi RSA private key (never commit) |

python3 ml/train.py --confidence_tau 0.60 --atr_min 15.0 --atr_max 30.0

python3 ml/train.py --confidence_tau 0.60 --atr_min 0 --atr_max 100 --calibrate_probabilities

python3 ml/train.py --calibrate_probabilities --optimize_threshold_for_profit --atr_min 0 --atr_max 100

python3 ml/train.py --calibrate_probabilities --optimize_threshold_for_profit --optimize_tau_by ev --atr_min 0 --atr_max 100

python3 ml/train.py --optimize_threshold_for_profit --optimize_tau_by ev --atr_min 0 --atr_max 100

python3 ml/train.py --atr_min 0 --atr_max 100 --confidence_tau 0.55 


Run this to optimize deadband and ATR
python3 deadband_atr_sweep.py \
    --csv data/btc_1min.csv \
    --horizon 15 \
    --cost_bps 1.0 \
    --heatmap \
    --output sweep_2d.csv


    Run this to train
    python3 ml/train.py \
    --optimize_threshold_for_profit \
    --optimize_tau_by ev \
    --atr_min 15.0 \
    --atr_max 30.0 \
    --deadband_bps 10.0 \
    --epochs 30