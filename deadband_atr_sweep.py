#!/usr/bin/env python3
"""
deadband_atr_sweep.py
=====================
2D grid search over deadband_bps × ATR channel to find the jointly
optimal training configuration for your 15-min BTC prediction model.

Uses calibrated LogisticRegression as a fast proxy (same RobustScaler,
mutual-info feature selection, and 70/15/15 temporal split as train.py).

Usage:
    python3 deadband_atr_sweep.py --csv data/btc_1min.csv

Optional flags:
    --horizon    15           Forward horizon in minutes (default: 15)
    --cost_bps   1.0          Transaction cost assumption in bps
    --top_k      64           Features for MI selection
    --deadbands  "2,4,6,8,10,12,15"
    --atr_ranges "0-100,10-40,15-30,15-25,20-35,10-25,20-40"
                              Comma-separated "min-max" ATR channel pairs
    --output     sweep_2d.csv Save full results table to CSV
    --heatmap                 Save EV/bar and accuracy heatmap PNGs
"""

import argparse
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# ATR-14 (Wilder EMA) — matches production_bot._calculate_live_atr_14
# ─────────────────────────────────────────────────────────────────────────────
def compute_atr14(df: pd.DataFrame) -> pd.Series:
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift(1)).abs(),
        (lo - cl.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — mirrors BTCMinuteProcessor.engineer_features
# (runs once on the full dataset; ATR gate applied per-cell)
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]

    df["return_1"]     = c.pct_change(1)
    df["return_5"]     = c.pct_change(5)
    df["return_15"]    = c.pct_change(15)
    df["return_30"]    = c.pct_change(30)
    df["log_return_1"] = np.log(c / c.shift(1))

    df["ma_5"]           = c.shift(1).rolling(5).mean()
    df["ma_15"]          = c.shift(1).rolling(15).mean()
    df["ma_30"]          = c.shift(1).rolling(30).mean()
    df["price_vs_ma_5"]  = (c - df["ma_5"])  / df["ma_5"].replace(0, np.nan)
    df["price_vs_ma_15"] = (c - df["ma_15"]) / df["ma_15"].replace(0, np.nan)

    df["vol_15"] = df["return_1"].rolling(15).std()
    df["vol_30"] = df["return_1"].rolling(30).std()
    df["ATR_14"] = compute_atr14(df)

    # RSI-14 (Wilder)
    delta    = c.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    df["rsi_14"] = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))

    if "volume" in df.columns:
        df["volume_ma_15"] = df["volume"].shift(1).rolling(15).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma_15"].replace(0, np.nan)
        typical = (df["high"] + df["low"] + c) / 3
        df["vwap_15"]       = ((typical * df["volume"]).rolling(15).sum() /
                                df["volume"].rolling(15).sum().replace(0, np.nan))
        df["price_vs_vwap"] = (c - df["vwap_15"]) / df["vwap_15"].replace(0, np.nan)

    if "cvd" in df.columns:
        df["cvd_delta_1"]  = df["cvd"].diff(1)
        df["cvd_delta_5"]  = df["cvd"].diff(5)
        df["cvd_ma_15"]    = df["cvd"].shift(1).rolling(15).mean()
        df["cvd_vs_ma_15"] = df["cvd"] - df["cvd_ma_15"]

    if {"buy_volume", "sell_volume"}.issubset(df.columns):
        total = (df["buy_volume"] + df["sell_volume"]).replace(0, np.nan)
        df["taker_buy_ratio"]       = df["buy_volume"] / total
        df["taker_buy_ratio_ma_15"] = df["taker_buy_ratio"].shift(1).rolling(15).mean()

    if {"spread_bps_open", "spread_bps_close"}.issubset(df.columns):
        df["spread_delta"] = df["spread_bps_close"] - df["spread_bps_open"]
        df["spread_ma_15"] = df["spread_bps_close"].shift(1).rolling(15).mean()
        df["spread_vs_ma"] = df["spread_bps_close"] - df["spread_ma_15"]

    if "spread_bps_max" in df.columns and "spread_bps_close" in df.columns:
        df["spread_spike"]       = df["spread_bps_max"] / df["spread_bps_close"].replace(0, np.nan)
        df["spread_spike_ma_15"] = df["spread_spike"].shift(1).rolling(15).mean()

    for snap in ("open", "close"):
        b, a = f"bid_depth_10_{snap}", f"ask_depth_10_{snap}"
        if {b, a}.issubset(df.columns):
            df[f"depth_ratio_{snap}"] = df[b] / (df[b] + df[a]).replace(0, np.nan)

    # Imbalance — correct column names (l1/l5/l10, not bare imbalance_open)
    for lvl in ("l1", "l5", "l10"):
        oc, cc = f"imbalance_{lvl}_open", f"imbalance_{lvl}_close"
        if {oc, cc}.issubset(df.columns):
            df[f"imbalance_{lvl}_delta"]    = df[cc] - df[oc]
            df[f"imbalance_{lvl}_ma_15"]    = df[cc].shift(1).rolling(15).mean()
            df[f"imbalance_{lvl}_momentum"] = df[cc] - df[f"imbalance_{lvl}_ma_15"]

    df["high_vol_flag"] = (df["vol_30"] > df["vol_30"].rolling(60).median()).astype(float)

    ts = df["timestamp"]
    df["hour"]           = ts.dt.hour
    df["minute_of_hour"] = ts.dt.minute
    df["day_of_week"]    = ts.dt.dayofweek
    df["session_asia"]   = ((ts.dt.hour >= 22) | (ts.dt.hour < 8)).astype(float)
    df["session_europe"] = ((ts.dt.hour >= 7)  & (ts.dt.hour < 16)).astype(float)
    df["session_us"]     = ((ts.dt.hour >= 13) & (ts.dt.hour < 21)).astype(float)

    # Mid price
    if {"best_bid_close", "best_ask_close"}.issubset(df.columns):
        ob_mid = (df["best_bid_close"] + df["best_ask_close"]) / 2
        df["mid_price"] = np.where(ob_mid.notna() & (ob_mid > 0), ob_mid,
                                   (df["high"] + df["low"]) / 2)
    else:
        df["mid_price"] = (df["high"] + df["low"]) / 2

    _mid_open  = ((df["best_bid_open"]  + df["best_ask_open"])  / 2).replace(0, np.nan) \
                 if {"best_bid_open", "best_ask_open"}.issubset(df.columns) else None
    _mid_close = df["mid_price"].replace(0, np.nan)

    # Bybit features
    if {"bybit_mark_open", "bybit_mark_close"}.issubset(df.columns):
        if _mid_open is not None:
            df["bybit_basis_open"]  = (_mid_open  - df["bybit_mark_open"])  / _mid_open
        df["bybit_basis_close"] = (_mid_close - df["bybit_mark_close"]) / _mid_close
        df["bybit_basis_delta"] = df["bybit_basis_close"] - df.get("bybit_basis_open",
                                                                    pd.Series(0, index=df.index))
        df["bybit_basis_ma_15"] = df["bybit_basis_close"].shift(1).rolling(15).mean()

    if "bybit_funding_rate" in df.columns:
        df["bybit_funding_ma_15"] = df["bybit_funding_rate"].shift(1).rolling(15).mean()

    if "bybit_oi" in df.columns:
        df["bybit_oi_norm"]         = df["bybit_oi"] / _mid_close
        df["bybit_oi_norm_delta_1"] = df["bybit_oi_norm"].diff(1)
        df["bybit_oi_norm_delta_5"] = df["bybit_oi_norm"].diff(5)

    if {"bybit_imbal_l10_open", "bybit_imbal_l10_close"}.issubset(df.columns):
        df["bybit_imbal_delta"] = df["bybit_imbal_l10_close"] - df["bybit_imbal_l10_open"]

    if "bybit_imbal_slope_open" in df.columns and "bybit_imbal_slope_close" in df.columns:
        df["bybit_imbal_slope_delta"] = (df["bybit_imbal_slope_close"] -
                                         df["bybit_imbal_slope_open"])

    if {"bybit_buy_volume", "bybit_sell_volume"}.issubset(df.columns):
        fv = (df["bybit_buy_volume"] + df["bybit_sell_volume"]).replace(0, np.nan)
        df["bybit_taker_buy_ratio"]       = df["bybit_buy_volume"] / fv
        df["bybit_taker_buy_ratio_ma_15"] = df["bybit_taker_buy_ratio"].shift(1).rolling(15).mean()

    if "bybit_cvd" in df.columns:
        df["bybit_cvd_delta_1"]  = df["bybit_cvd"].diff(1)
        df["bybit_cvd_delta_5"]  = df["bybit_cvd"].diff(5)
        df["bybit_cvd_ma_15"]    = df["bybit_cvd"].shift(1).rolling(15).mean()
        df["bybit_cvd_vs_ma_15"] = df["bybit_cvd"] - df["bybit_cvd_ma_15"]

    if {"cvd", "bybit_cvd"}.issubset(df.columns):
        _cs = df["cvd"].rolling(60).std().replace(0, np.nan)
        _bs = df["bybit_cvd"].rolling(60).std().replace(0, np.nan)
        df["cvd_divergence"] = (df["cvd"] / _cs) - (df["bybit_cvd"] / _bs)

    if {"bybit_liq_long_vol", "bybit_liq_short_vol"}.issubset(df.columns):
        df["bybit_liq_net"]   = df["bybit_liq_long_vol"] - df["bybit_liq_short_vol"]
        df["bybit_liq_total"] = df["bybit_liq_long_vol"] + df["bybit_liq_short_vol"]
        if "volume" in df.columns:
            _v = df["volume"].replace(0, np.nan)
            df["bybit_liq_long_ratio"]  = df["bybit_liq_long_vol"]  / _v
            df["bybit_liq_short_ratio"] = df["bybit_liq_short_vol"] / _v

    if {"bybit_bar_high", "bybit_bar_low"}.issubset(df.columns):
        df["bybit_bar_range"]     = (df["bybit_bar_high"] - df["bybit_bar_low"]) / _mid_close
        _sr = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        df["bybit_range_vs_spot"] = df["bybit_bar_range"] / _sr.replace(0, np.nan)

    if "bybit_prev_price_1h" in df.columns and "bybit_mark_close" in df.columns:
        df["bybit_1h_momentum"] = ((df["bybit_mark_close"] - df["bybit_prev_price_1h"]) /
                                    df["bybit_prev_price_1h"].replace(0, np.nan))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Columns excluded from features
# ─────────────────────────────────────────────────────────────────────────────
EXCLUDE = {
    "symbol", "timestamp", "date", "future_mid", "ret", "ret_bps",
    "y", "y_binary", "y_direction", "mid_price", "ATR_14",
    "best_bid_open", "best_ask_open", "best_bid_close", "best_ask_close",
    "open", "high", "low", "close",
}


# ─────────────────────────────────────────────────────────────────────────────
# Single cell evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_cell(
    feat_df: pd.DataFrame,
    horizon: int,
    deadband: float,
    atr_min: float,
    atr_max: float,
    top_k: int,
    cost_bps: float,
    min_train: int = 1_000,
) -> dict:

    base = {"deadband_bps": deadband, "atr_min": atr_min, "atr_max": atr_max}

    # ATR gate
    if "ATR_14" in feat_df.columns:
        gated = feat_df[feat_df["ATR_14"].between(atr_min, atr_max)].copy()
    else:
        gated = feat_df.copy()

    if len(gated) < 100:
        return {**base, "note": "atr_too_restrictive", "n_labeled": len(gated)}

    # Forward returns and deadband filter
    mid        = gated["mid_price"]
    future_mid = mid.shift(-horizon)
    ret_bps    = (future_mid / mid - 1) * 10000
    gated      = gated.copy()
    gated["ret_bps"] = ret_bps
    gated      = gated.dropna(subset=["ret_bps"])
    gated      = gated[gated["ret_bps"].abs() > deadband].copy()
    n_labeled  = len(gated)

    if n_labeled < 500:
        return {**base, "note": "too_few_labeled", "n_labeled": n_labeled}

    gated["y_direction"] = (gated["ret_bps"] > 0).astype(int)

    # Temporal 70/15/15 split
    n       = len(gated)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    tr = gated.iloc[:n_train]
    va = gated.iloc[n_train : n_train + n_val]
    te = gated.iloc[n_train + n_val :]

    if len(tr) < min_train:
        return {**base, "note": "train_too_small", "n_labeled": n_labeled,
                "n_train": len(tr)}

    # Feature prep
    num_cols  = gated.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in EXCLUDE and gated[c].nunique() > 1]

    def prep(df_):
        return df_[feat_cols].replace([np.inf, -np.inf], np.nan)

    X_tr, y_tr = prep(tr), tr["y_direction"].values
    X_va, y_va = prep(va), va["y_direction"].values
    X_te, y_te = prep(te), te["y_direction"].values
    ret_te     = te["ret_bps"].values

    # Impute
    imp      = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)
    X_va_imp = imp.transform(X_va)
    X_te_imp = imp.transform(X_te)

    # MI feature selection (subsample for speed)
    k      = min(top_k, X_tr_imp.shape[1])
    n_mi   = min(30_000, len(X_tr_imp))
    idx    = np.random.default_rng(42).choice(len(X_tr_imp), n_mi, replace=False)
    sel    = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, n_neighbors=2, random_state=42),
        k=k,
    )
    sel.fit(X_tr_imp[idx], y_tr[idx])
    X_tr_s = sel.transform(X_tr_imp)
    X_va_s = sel.transform(X_va_imp)
    X_te_s = sel.transform(X_te_imp)

    # Scale
    sc     = RobustScaler()
    X_tr_s = sc.fit_transform(X_tr_s)
    X_va_s = sc.transform(X_va_s)
    X_te_s = sc.transform(X_te_s)

    # Train LR + calibrate on val
    lr  = LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced", random_state=42)
    lr.fit(X_tr_s, y_tr)
    cal = CalibratedClassifierCV(FrozenEstimator(lr), method="isotonic")
    cal.fit(X_va_s, y_va)

    # Evaluate on test — grid-search tau
    proba    = cal.predict_proba(X_te_s)
    p_up     = proba[:, 1]
    conf     = np.maximum(p_up, 1 - p_up)
    dir_pred = (p_up > 0.5).astype(int)
    val_acc  = (dir_pred == y_te).mean()

    best_ev = best_profit = best_wr = -999.0
    best_tau = 0.50
    for tau in np.arange(0.50, 0.91, 0.01):
        mask = conf >= tau
        if mask.sum() < 10:
            continue
        gross = np.where(dir_pred[mask] == 1, ret_te[mask], -ret_te[mask])
        net   = gross - cost_bps
        ev    = net.mean() * mask.mean()
        if ev > best_ev:
            best_ev     = ev
            best_tau    = tau
            best_profit = net.mean()
            best_wr     = (net > 0).mean()

    # ATR distribution within this gate
    atr_vals = feat_df["ATR_14"] if "ATR_14" in feat_df.columns else pd.Series([np.nan])
    gated_atr = atr_vals[atr_vals.between(atr_min, atr_max)]

    return {
        **base,
        "note":           "ok",
        "n_atr_eligible": len(gated_atr),
        "atr_pct":        round(len(gated_atr) / len(feat_df) * 100, 1),
        "n_labeled":      n_labeled,
        "coverage_pct":   round(n_labeled / len(feat_df) * 100, 1),
        "up_pct":         round(gated["y_direction"].mean() * 100, 1),
        "n_train":        len(tr),
        "val_accuracy":   round(val_acc * 100, 2),
        "best_tau":       round(best_tau, 2),
        "val_win_rate":   round(best_wr * 100, 2),
        "avg_profit_bps": round(best_profit, 2),
        "ev_per_obs":     round(best_ev, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_heatmaps(results_df: pd.DataFrame, output_stem: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  (matplotlib not available — skipping heatmaps)")
        return

    ok = results_df[results_df["note"] == "ok"].copy()
    if ok.empty:
        return

    ok["atr_label"] = ok["atr_min"].astype(int).astype(str) + "–" + ok["atr_max"].astype(int).astype(str)

    for metric, title, fmt in [
        ("ev_per_obs",     "EV per bar",      ".2f"),
        ("val_accuracy",   "Val Accuracy (%)", ".1f"),
        ("avg_profit_bps", "Avg Profit (bps)", ".1f"),
        ("n_train",        "Train samples",    ".0f"),
    ]:
        pivot = ok.pivot_table(index="atr_label", columns="deadband_bps",
                               values=metric, aggfunc="mean")

        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.1),
                                        max(4, len(pivot.index) * 0.7)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
        plt.colorbar(im, ax=ax, label=title)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.0f}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Deadband (bps)")
        ax.set_ylabel("ATR Range (USD)")
        ax.set_title(f"{title} — Deadband × ATR Grid")

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                            fontsize=8, color="black")

        plt.tight_layout()
        fname = f"{output_stem}_{metric}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved heatmap: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       required=True)
    parser.add_argument("--horizon",   type=int,   default=15)
    parser.add_argument("--cost_bps",  type=float, default=1.0)
    parser.add_argument("--top_k",     type=int,   default=64)
    parser.add_argument("--deadbands", type=str,
                        default="2,4,6,8,10,12,15")
    parser.add_argument("--atr_ranges", type=str,
                        default="0-100,10-40,15-30,15-25,20-35,10-25,20-40")
    parser.add_argument("--output",    type=str,   default="sweep_2d.csv")
    parser.add_argument("--heatmap",   action="store_true")
    args = parser.parse_args()

    deadbands = [float(x) for x in args.deadbands.split(",")]
    atr_ranges = []
    for pair in args.atr_ranges.split(","):
        lo, hi = pair.strip().split("-")
        atr_ranges.append((float(lo), float(hi)))

    n_cells = len(deadbands) * len(atr_ranges)
    print(f"\n{'='*72}")
    print(f"  2D DEADBAND × ATR SWEEP")
    print(f"  CSV: {Path(args.csv).name}  |  horizon={args.horizon}min  |  cost={args.cost_bps}bps")
    print(f"  Deadbands:  {deadbands}")
    print(f"  ATR ranges: {atr_ranges}")
    print(f"  Total cells: {n_cells}")
    print(f"{'='*72}\n")

    # ── Load & engineer features once ────────────────────────────────────────
    print("Loading CSV...", flush=True)
    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} rows  |  {df['timestamp'].min()} → {df['timestamp'].max()}")

    print("Engineering features (runs once)...", flush=True)
    t0 = time.perf_counter()
    feat_df = engineer_features(df)
    print(f"  Done in {time.perf_counter()-t0:.1f}s  |  {feat_df.shape[1]} columns\n")

    # ── ATR distribution summary ──────────────────────────────────────────────
    if "ATR_14" in feat_df.columns:
        atr = feat_df["ATR_14"].dropna()
        print("  ATR-14 distribution across full dataset:")
        for p in [5, 10, 25, 50, 75, 90, 95]:
            print(f"    p{p:2d}: {np.percentile(atr, p):.1f} USD")
        print()

    # ── 15-min return distribution (no gate) ─────────────────────────────────
    mid      = feat_df["mid_price"]
    fwd_ret  = ((mid.shift(-args.horizon) / mid - 1) * 10000).dropna().abs()
    print("  15-min |return| distribution (full dataset):")
    for p in [10, 25, 50, 75, 90]:
        print(f"    p{p:2d}: {np.percentile(fwd_ret, p):.1f} bps")
    print(f"    mean: {fwd_ret.mean():.1f} bps  std: {fwd_ret.std():.1f} bps\n")

    # ── Grid sweep ────────────────────────────────────────────────────────────
    results = []
    cell_num = 0
    t_start = time.perf_counter()

    for (atr_min, atr_max), db in product(atr_ranges, deadbands):
        cell_num += 1
        elapsed  = time.perf_counter() - t_start
        eta      = (elapsed / cell_num) * (n_cells - cell_num) if cell_num > 1 else 0
        print(f"  [{cell_num:>2}/{n_cells}]  ATR={atr_min:.0f}-{atr_max:.0f}  "
              f"DB={db:.0f}  (ETA {eta:.0f}s)", flush=True, end="  ")

        r = evaluate_cell(
            feat_df,
            horizon=args.horizon,
            deadband=db,
            atr_min=atr_min,
            atr_max=atr_max,
            top_k=args.top_k,
            cost_bps=args.cost_bps,
        )
        results.append(r)

        if r["note"] == "ok":
            print(f"acc={r['val_accuracy']:.1f}%  EV={r['ev_per_obs']:.3f}  "
                  f"N_train={r['n_train']:,}  cov={r['coverage_pct']:.1f}%")
        else:
            print(f"SKIP: {r['note']}")

    # ── Results table ─────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    ok         = results_df[results_df["note"] == "ok"].copy()

    print(f"\n{'='*72}")
    print(f"  RESULTS SUMMARY  (sorted by EV/bar, top 15)")
    print(f"{'='*72}")
    print(f"{'ATR range':>10} | {'DB':>4} | {'N_tr':>6} | {'Cov%':>5} | "
          f"{'Acc%':>6} | {'WR%':>6} | {'Profit':>7} | {'EV/bar':>7}")
    print("-" * 72)

    if not ok.empty:
        top = ok.nlargest(15, "ev_per_obs")
        for _, r in top.iterrows():
            print(f"{r['atr_min']:.0f}–{r['atr_max']:.0f}  "
                  f"{'':>3} | {r['deadband_bps']:>4.0f} | {r['n_train']:>6,} | "
                  f"{r['coverage_pct']:>5.1f} | {r['val_accuracy']:>6.2f} | "
                  f"{r['val_win_rate']:>6.2f} | {r['avg_profit_bps']:>7.2f} | "
                  f"{r['ev_per_obs']:>7.4f}")

    # ── Pivot: EV/bar heatmap in text ─────────────────────────────────────────
    if not ok.empty:
        ok["atr_label"] = (ok["atr_min"].astype(int).astype(str) + "–" +
                           ok["atr_max"].astype(int).astype(str))
        pivot_ev  = ok.pivot_table(index="atr_label", columns="deadband_bps",
                                   values="ev_per_obs", aggfunc="mean")
        pivot_acc = ok.pivot_table(index="atr_label", columns="deadband_bps",
                                   values="val_accuracy", aggfunc="mean")
        pivot_n   = ok.pivot_table(index="atr_label", columns="deadband_bps",
                                   values="n_train", aggfunc="mean")

        print(f"\n  EV/bar heatmap  (rows=ATR range, cols=deadband):")
        print("  " + "  ".join(f"{c:>6.0f}" for c in pivot_ev.columns) + "  ← deadband bps")
        for idx_label, row in pivot_ev.iterrows():
            vals = "  ".join(
                f"{v:>6.3f}" if not np.isnan(v) else "   n/a"
                for v in row.values
            )
            print(f"  {idx_label:<8}  {vals}")

        print(f"\n  Val accuracy heatmap (%):")
        print("  " + "  ".join(f"{c:>6.0f}" for c in pivot_acc.columns) + "  ← deadband bps")
        for idx_label, row in pivot_acc.iterrows():
            vals = "  ".join(
                f"{v:>6.1f}" if not np.isnan(v) else "   n/a"
                for v in row.values
            )
            print(f"  {idx_label:<8}  {vals}")

        print(f"\n  Train-sample count heatmap:")
        print("  " + "  ".join(f"{c:>6.0f}" for c in pivot_n.columns) + "  ← deadband bps")
        for idx_label, row in pivot_n.iterrows():
            vals = "  ".join(
                f"{int(v):>6}" if not np.isnan(v) else "   n/a"
                for v in row.values
            )
            print(f"  {idx_label:<8}  {vals}")

        # ── Best cell ──────────────────────────────────────────────────────────
        best = ok.loc[ok["ev_per_obs"].idxmax()]
        best_acc = ok.loc[ok["val_accuracy"].idxmax()]

        # Filter for cells with adequate training samples (>= 1500)
        safe = ok[ok["n_train"] >= 1_500]
        best_safe = safe.loc[safe["ev_per_obs"].idxmax()] if not safe.empty else best

        print(f"\n{'='*72}")
        print(f"  RECOMMENDATION")
        print(f"{'='*72}")
        print(f"  Best EV/bar (all):     ATR={best['atr_min']:.0f}–{best['atr_max']:.0f}  "
              f"DB={best['deadband_bps']:.0f} bps  "
              f"(EV={best['ev_per_obs']:.4f}, acc={best['val_accuracy']:.1f}%, "
              f"N_train={int(best['n_train']):,})")
        if not safe.empty:
            print(f"  Best EV (N≥1500):      ATR={best_safe['atr_min']:.0f}–{best_safe['atr_max']:.0f}  "
                  f"DB={best_safe['deadband_bps']:.0f} bps  "
                  f"(EV={best_safe['ev_per_obs']:.4f}, acc={best_safe['val_accuracy']:.1f}%, "
                  f"N_train={int(best_safe['n_train']):,})")
        print(f"  Best accuracy:         ATR={best_acc['atr_min']:.0f}–{best_acc['atr_max']:.0f}  "
              f"DB={best_acc['deadband_bps']:.0f} bps  "
              f"(acc={best_acc['val_accuracy']:.1f}%, "
              f"EV={best_acc['ev_per_obs']:.4f}, N_train={int(best_acc['n_train']):,})")
        print()
        print(f"  Use 'Best EV (N≥1500)' as your primary pick — it balances signal")
        print(f"  quality against training data volume with a floor on sample size.")
        print()
        print(f"  Run train.py with:")
        print(f"    --deadband {best_safe['deadband_bps']:.0f} "
              f"--atr_min {best_safe['atr_min']:.0f} "
              f"--atr_max {best_safe['atr_max']:.0f}")
        print(f"{'='*72}\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"  Full results saved to: {args.output}")

    # ── Heatmap PNGs ─────────────────────────────────────────────────────────
    if args.heatmap and not ok.empty:
        stem = str(Path(args.output).stem) if args.output else "sweep_2d"
        save_heatmaps(ok, stem)


if __name__ == "__main__":
    main()