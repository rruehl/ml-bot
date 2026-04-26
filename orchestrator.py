#!/usr/bin/env python3
"""
ML Trading System Orchestrator
===============================
Starts and supervises all system components as persistent subprocesses:
  - collector/btc_ws_collector.py  (24/7 data collection)
  - bot/production_bot.py          (24/7 live trading)
  - bot/dashboard.py               (24/7 monitoring UI)
  - bot/drift_monitor.py           (concept drift detection + auto-retrain)

Schedules weekly model retraining every Sunday at midnight CT:
  preprocess.py → train.py → artifacts/model_updated.flag
  The bot's model_watchdog() picks up the flag and hot-reloads artifacts.

Usage:
  python3 orchestrator.py

Stop with Ctrl+C or SIGTERM — all subprocesses are cleanly terminated.
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler

PROJECT_ROOT  = Path(__file__).resolve().parent
VENV_PYTHON   = PROJECT_ROOT / "venv" / "bin" / "python3"
LOG_FILE      = PROJECT_ROOT / "logs" / "orchestrator.log"

COLLECTOR_CMD      = [str(VENV_PYTHON), "collector/btc_ws_collector.py"]
BOT_CMD            = [str(VENV_PYTHON), "bot/production_bot.py"]
DASHBOARD_CMD      = [str(VENV_PYTHON), "bot/dashboard.py"]
DRIFT_MONITOR_CMD  = [str(VENV_PYTHON), "bot/drift_monitor.py"]
PREPROCESS_CMD = [str(VENV_PYTHON), "ml/preprocess.py",
                  "--input",  "data/btc_1min.csv",
                  "--output", "data/processed/"]
TRAIN_CMD      = [str(VENV_PYTHON), "ml/train.py",
                  "--confidence_tau", "0.60",
                  "--atr_min",        "15.0",
                  "--atr_max",        "30.0",
                  "--horizon_min",    "15",
                  "--top_k_features", "128",
                  "--epochs",         "30"]


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("orchestrator")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(str(LOG_FILE), maxBytes=10_000_000, backupCount=5)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


log = _setup_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Managed subprocess
# ─────────────────────────────────────────────────────────────────────────────

class ManagedProcess:
    def __init__(self, name: str, cmd: list, restart_delay: float = 10.0):
        self.name          = name
        self.cmd           = cmd
        self.restart_delay = restart_delay
        self.proc: asyncio.subprocess.Process | None = None
        self._running      = True
        self.restart_count = 0
        self.started_at    = 0.0

    async def supervise(self) -> None:
        """Start the process and restart it whenever it exits."""
        while self._running:
            log.info(f"[{self.name}] Starting (restart #{self.restart_count})")
            self.started_at = time.time()
            try:
                self.proc = await asyncio.create_subprocess_exec(
                    *self.cmd,
                    cwd=str(PROJECT_ROOT),
                    # stdout/stderr inherit from the orchestrator so they appear
                    # in the terminal and in any parent process log capture.
                )
                returncode = await self.proc.wait()
                uptime = time.time() - self.started_at
                log.warning(
                    f"[{self.name}] Exited (code={returncode}, "
                    f"uptime={uptime:.0f}s) — restarting in {self.restart_delay}s"
                )
            except Exception as exc:
                log.error(f"[{self.name}] Failed to start: {exc}")

            self.restart_count += 1
            if self._running:
                await asyncio.sleep(self.restart_delay)

        log.info(f"[{self.name}] Supervisor stopped.")

    async def stop(self) -> None:
        """Gracefully terminate the subprocess."""
        self._running = False
        if self.proc and self.proc.returncode is None:
            log.info(f"[{self.name}] Sending SIGTERM ...")
            try:
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), timeout=10.0)
                log.info(f"[{self.name}] Stopped cleanly.")
            except asyncio.TimeoutError:
                log.warning(f"[{self.name}] SIGTERM timeout — sending SIGKILL")
                self.proc.kill()
            except Exception as exc:
                log.error(f"[{self.name}] Error during stop: {exc}")

    @property
    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None


# ─────────────────────────────────────────────────────────────────────────────
# Retraining pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def run_retraining_pipeline() -> None:
    """
    Runs the full weekly retraining pipeline:
      1. preprocess.py  — rebuilds processed parquet from raw CSV
      2. train.py       — trains model; writes model_updated.flag on success

    The bot's model_watchdog coroutine detects the flag and hot-reloads
    artifacts without restarting the bot.

    Called automatically every Sunday at midnight CT by APScheduler.
    Can also be triggered manually for testing.
    """
    t0 = time.time()
    log.info("=" * 60)
    log.info("RETRAINING PIPELINE: starting")
    log.info("=" * 60)

    # Step 1: Preprocess
    log.info("RETRAINING [1/2] Running preprocess.py ...")
    try:
        proc = await asyncio.create_subprocess_exec(
            *PREPROCESS_CMD,
            cwd=str(PROJECT_ROOT),
        )
        rc = await proc.wait()
    except Exception as exc:
        log.error(f"RETRAINING [1/2] Failed to launch preprocess.py: {exc}")
        return

    if rc != 0:
        log.error(f"RETRAINING [1/2] preprocess.py exited with code {rc} — aborting pipeline")
        return

    elapsed = time.time() - t0
    log.info(f"RETRAINING [1/2] preprocess.py completed in {elapsed:.0f}s")

    # Step 2: Train
    log.info("RETRAINING [2/2] Running train.py ...")
    try:
        proc = await asyncio.create_subprocess_exec(
            *TRAIN_CMD,
            cwd=str(PROJECT_ROOT),
        )
        rc = await proc.wait()
    except Exception as exc:
        log.error(f"RETRAINING [2/2] Failed to launch train.py: {exc}")
        return

    if rc != 0:
        log.error(f"RETRAINING [2/2] train.py exited with code {rc} — model NOT updated")
        return

    total = time.time() - t0
    log.info(f"RETRAINING [2/2] train.py completed. Total pipeline: {total:.0f}s")
    log.info("RETRAINING PIPELINE: complete — bot will hot-reload within 5 min")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("=" * 60)
    log.info("ML Trading System Orchestrator starting")
    log.info(f"Project root: {PROJECT_ROOT}")
    log.info(f"Python:       {VENV_PYTHON}")
    log.info("=" * 60)

    collector      = ManagedProcess("collector",      COLLECTOR_CMD,     restart_delay=10.0)
    bot            = ManagedProcess("bot",            BOT_CMD,           restart_delay=15.0)
    dashboard      = ManagedProcess("dashboard",      DASHBOARD_CMD,     restart_delay=10.0)
    drift_monitor  = ManagedProcess("drift_monitor",  DRIFT_MONITOR_CMD, restart_delay=10.0)
    processes      = [collector, bot, dashboard, drift_monitor]

    # Weekly retraining: Sunday 00:00 CT
    scheduler = AsyncIOScheduler(timezone="America/Chicago")
    scheduler.add_job(
        run_retraining_pipeline,
        trigger="cron",
        day_of_week="sun",
        hour=0,
        minute=0,
        id="weekly_retrain",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()

    next_run = scheduler.get_job("weekly_retrain").next_run_time
    log.info(f"Retraining scheduler started — next run: {next_run}")

    # Graceful shutdown on SIGINT / SIGTERM
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _handle_signal():
        log.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    # Start all supervisors as concurrent tasks
    tasks = [
        asyncio.create_task(collector.supervise(),      name="supervisor-collector"),
        asyncio.create_task(bot.supervise(),            name="supervisor-bot"),
        asyncio.create_task(dashboard.supervise(),      name="supervisor-dashboard"),
        asyncio.create_task(drift_monitor.supervise(),  name="supervisor-drift"),
    ]

    # Wait until a shutdown signal arrives
    await shutdown_event.wait()

    log.info("Stopping all processes ...")
    for p in processes:
        await p.stop()

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    scheduler.shutdown(wait=False)
    log.info("Orchestrator stopped.")


if __name__ == "__main__":
    asyncio.run(main())
