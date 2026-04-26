#!/usr/bin/env python3
"""
Drift Monitor
=============
Watches the production trade log for concept drift by computing a rolling
win rate over the last 50 PAYOUT/SETTLE events. If the win rate drops below
0.53, triggers an emergency retrain of the ML model (subject to a 12-hour
cooldown to prevent thrashing).

Runs as a persistent subprocess managed by orchestrator.py.
"""

import asyncio
import logging
import subprocess
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_CSV      = PROJECT_ROOT / "logs" / "production_log.csv"
BOT_LOG      = PROJECT_ROOT / "logs" / "bot.log"

WIN_RATE_THRESHOLD = 0.53
WINDOW_SIZE        = 50
COOLDOWN_HOURS     = 12
CHECK_INTERVAL_SEC = 300  # 5 minutes

# sys.executable is the venv Python that started this process, matching the
# environment used by orchestrator.py for all other subprocesses.
RETRAIN_CMD = [
    sys.executable, "ml/train.py",
    "--optimize_threshold_for_profit",
    "--optimize_tau_by", "ev",
    "--atr_min",         "15.0",
    "--atr_max",         "30.0",
]


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("drift_monitor")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  [drift_monitor]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    BOT_LOG.parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(str(BOT_LOG), maxBytes=10_000_000, backupCount=5)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = _setup_logger()


async def monitor_loop() -> None:
    last_retrain_ts: float = 0.0

    while True:
        await asyncio.sleep(CHECK_INTERVAL_SEC)

        try:
            if not LOG_CSV.exists():
                continue

            df = pd.read_csv(LOG_CSV, low_memory=False)
            trades = df[df["event"].isin({"PAYOUT", "SETTLE"})].copy()

            if len(trades) < WINDOW_SIZE:
                log.info(
                    f"Only {len(trades)} trade outcomes recorded — "
                    f"need {WINDOW_SIZE} before drift evaluation begins"
                )
                continue

            window   = trades.tail(WINDOW_SIZE)
            wins     = int((window["event"] == "PAYOUT").sum())
            win_rate = wins / WINDOW_SIZE

            log.info(
                f"Rolling win rate (last {WINDOW_SIZE} trades): "
                f"{win_rate:.3f}  ({wins}/{WINDOW_SIZE})"
            )

            if win_rate >= WIN_RATE_THRESHOLD:
                continue

            # Win rate below threshold — check cooldown before triggering retrain
            elapsed   = time.time() - last_retrain_ts
            cooldown  = COOLDOWN_HOURS * 3600
            if elapsed < cooldown:
                remaining_h = (cooldown - elapsed) / 3600
                log.info(
                    f"Drift detected (win_rate={win_rate:.3f}) but cooldown active — "
                    f"{remaining_h:.1f}h until next retrain is allowed"
                )
                continue

            log.warning(
                f"DRIFT DETECTED: win_rate={win_rate:.3f} < {WIN_RATE_THRESHOLD} "
                f"over last {WINDOW_SIZE} trades — triggering emergency retrain"
            )

            t0 = time.time()
            result = subprocess.run(RETRAIN_CMD, cwd=str(PROJECT_ROOT))
            duration = time.time() - t0
            last_retrain_ts = time.time()

            if result.returncode == 0:
                log.info(f"Drift retrain completed successfully in {duration:.0f}s")
            else:
                log.error(
                    f"Drift retrain exited with code {result.returncode} "
                    f"after {duration:.0f}s — model NOT updated"
                )

        except Exception as exc:
            log.error(f"Monitor loop error: {exc}")


if __name__ == "__main__":
    log.info(
        f"Drift monitor started  "
        f"window={WINDOW_SIZE}  threshold={WIN_RATE_THRESHOLD}  "
        f"cooldown={COOLDOWN_HOURS}h  interval={CHECK_INTERVAL_SEC}s"
    )
    asyncio.run(monitor_loop())
