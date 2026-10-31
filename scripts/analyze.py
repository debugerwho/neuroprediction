#!/usr/bin/env python3
"""Empirical analysis: liquidation-map magnet levels vs realized BTC price action.

Inputs:
  .cache/parsed/{snapshots,levels,collisions}.csv  (from parse_snapshots.py)
  data/btc_1m/spot/BTCUSDT-1m-*.csv               (Binance Vision daily zips)

Outputs:
  .cache/analysis/targets.csv     — per (snapshot, window, side): strongest levels + fuel
  .cache/analysis/labels.csv      — touch labels per (snapshot, window, side, target)
  .cache/analysis/events.csv      — aggregated per-snapshot features for modeling
  console summary of empirical regularities
"""
import glob
import os
import numpy as np
import pandas as pd

PD = ".cache/parsed"
AD = ".cache/analysis"
os.makedirs(AD, exist_ok=True)

WINDOWS = ["1 день", "7 день", "30 день", "90 день", "180 день", "1 Год"]
HORIZONS = [15, 30, 60, 240]

# ---------- load klines ----------
def load_klines():
    frames = []
    cols = [
        "open_time_us", "open", "high", "low", "close", "volume",
        "close_time_us", "quote_vol", "trades", "taker_base", "taker_quote", "ignore",
    ]
    for f in sorted(glob.glob("data/btc_1m/spot/BTCUSDT-1m-*.csv")):
        df = pd.read_csv(f, header=None, names=cols)
        frames.append(df)
    k = pd.concat(frames, ignore_index=True)
    k["ts"] = pd.to_datetime((k["open_time_us"] // 1000).astype("int64"), unit="ms", utc=True)
    k = k.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    k = k[["ts", "open", "high", "low", "close", "volume", "quote_vol", "trades"]].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float,
         "quote_vol": float, "trades": float}
    )
    return k

k = load_klines()
k_idx = k.set_index("ts")

snap = pd.read_csv(f"{PD}/snapshots.csv")
lev = pd.read_csv(f"{PD}/levels.csv")
for df in (snap, lev):
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
try:
    coll = pd.read_csv(f"{PD}/collisions.csv")
    coll["ts_utc"] = pd.to_datetime(coll["ts_utc"], utc=True)
except Exception:
    coll = pd.DataFrame(columns=["file", "ts_utc", "window", "side", "level", "usd", "reason"])

snap = snap.drop_duplicates("ts_utc").sort_values("ts_utc").reset_index(drop=True)
print(f"klines={len(k)} snapshots={len(snap)} levels={len(lev)} collisions_rows={len(coll)}")
print(f"window: {snap.ts_utc.min()} .. {snap.ts_utc.max()}")

# ---------- per-snapshot targets ----------
# For each snapshot & window & side: reconstruct fuel profile; strongest level =
# level with max кум_коэфф within +-15 bins of the divider; also top-3 peaks.
rows = []
for (file, ts, window, side), g in lev.groupby(["file", "ts_utc", "window", "side"], sort=False):
    g = g.sort_values("level").reset_index(drop=True)
    rec = g["cum"].to_numpy()
    rloc = g["loc"].to_numpy()
    lvl = g["level"].to_numpy()
    # local peaks in кум
    peaks = []
    for i in range(1, len(rloc) - 1):
        if rloc[i] >= rloc[i - 1] and rloc[i] > rloc[i + 1] and rloc[i] > 1.0:
            peaks.append((lvl[i], rloc[i], rec[i]))
    peaks.sort(key=lambda x: -x[1])
    # nearest peak to divider
    div_col = {"1 день": "w1d", "7 день": "w7d", "30 день": "w30d",
               "90 день": "w90d", "180 день": "w180d", "1 Год": "w1y"}[window]
    srow = snap.loc[snap["file"] == file]
    div = srow[div_col].iloc[0] if len(srow) else np.nan
    mid = srow["mid"].iloc[0] if len(srow) else np.nan
    near = None
    if len(peaks) and not np.isnan(div):
        near = min(peaks, key=lambda p: abs(p[0] - div))
    rows.append({
        "file": file, "ts_utc": ts, "window": window, "side": side,
        "n_levels": len(lvl), "n_peaks": len(peaks),
        "max_kum": float(rloc.max()) if len(rloc) else np.nan,
        "max_kum_level": float(lvl[np.argmax(rloc)]) if len(rloc) else np.nan,
        "total_fuel": float(rec[-1]) if len(rec) else np.nan,  # cumulative at far end
        "near_peak_level": near[0] if near else np.nan,
        "near_peak_kum": near[1] if near else np.nan,
        "near_peak_fuel": near[2] if near else np.nan,
        "near_peak_dist": (near[0] - mid) if near else np.nan,
        "divider": div, "mid": mid,
    })
targets = pd.DataFrame(rows)
targets.to_csv(f"{AD}/targets.csv", index=False)
print("targets rows:", len(targets))

# ---------- touch labels ----------
k_high = k.set_index("ts")["high"]
k_low = k.set_index("ts")["low"]
k_close = k.set_index("ts")["close"]
times = k["ts"].to_numpy()

def touch_up(t0, level, H):
    """First time high >= level within (t0, t0+H min]; returns minutes or None."""
    t1 = t0 + pd.Timedelta(minutes=H)
    m = k[(k["ts"] > t0) & (k["ts"] <= t1)]
    hit = m[m["high"] >= level]
    if len(hit):
        return (hit["ts"].iloc[0] - t0).total_seconds() / 60.0
    return np.nan

def touch_down(t0, level, H):
    t1 = t0 + pd.Timedelta(minutes=H)
    m = k[(k["ts"] > t0) & (k["ts"] <= t1)]
    hit = m[m["low"] <= level]
    if len(hit):
        return (hit["ts"].iloc[0] - t0).total_seconds() / 60.0
    return np.nan

tg = targets.dropna(subset=["near_peak_level"]).copy()
lab_rows = []
for r in tg.itertuples():
    t0 = r.ts_utc
    lvl = r.near_peak_level
    for H in HORIZONS:
        if r.side == "UP":
            tt = touch_up(t0, lvl, H)
        else:
            tt = touch_down(t0, lvl, H)
        lab_rows.append({
            "file": r.file, "ts_utc": t0, "window": r.window, "side": r.side,
            "near_peak_level": lvl, "near_peak_kum": r.near_peak_kum,
            "near_peak_dist": r.near_peak_dist, "horizon": H,
            "touched": int(not np.isnan(tt)), "tt_min": tt,
        })
labels = pd.DataFrame(lab_rows)
labels.to_csv(f"{AD}/labels.csv", index=False)
print("labels rows:", len(labels))

# ---------- events (features) ----------
ev = []
for r in snap.itertuples():
    t0 = r.ts_utc
    sel = tg[tg["file"] == r.file]
    up = sel[sel["side"] == "UP"]
    dn = sel[sel["side"] == "DOWN"]
    up_near = up.iloc[0] if len(up) else None
    dn_near = dn.iloc[0] if len(dn) else None
    # fuel asymmetry on 1d window
    w1 = sel[sel["window"] == "1 день"]
    up_fuel1 = w1[w1["side"] == "UP"]["total_fuel"].sum()
    dn_fuel1 = w1[w1["side"] == "DOWN"]["total_fuel"].sum()
    # realized vol past 60m
    past = k[(k["ts"] > t0 - pd.Timedelta(minutes=60)) & (k["ts"] <= t0)]
    rv60 = past["close"].pct_change().std() * np.sqrt(60) if len(past) > 5 else np.nan
    # future 60m move
    fut = k[(k["ts"] > t0) & (k["ts"] <= t0 + pd.Timedelta(minutes=60))]
    fwd_ret = (fut["close"].iloc[-1] / r.mid - 1) if len(fut) else np.nan
    fwd_max_up = (fut["high"].max() / r.mid - 1) if len(fut) else np.nan
    fwd_max_dn = (r.mid / fut["low"].min() - 1) if len(fut) else np.nan
    ev.append({
        "file": r.file, "ts_utc": t0, "mid": r.mid, "spread": r.spread,
        "bid_count": r.bid_count, "ask_count": r.ask_count,
        "book_imb": (r.bid_count - r.ask_count) / (r.bid_count + r.ask_count),
        "dead_up": (up_near["divider"] - r.best_ask) if up_near is not None else np.nan,
        "dead_dn": (r.best_bid - dn_near["divider"]) if dn_near is not None else np.nan,
        "up_dist": up_near["near_peak_dist"] if up_near is not None else np.nan,
        "dn_dist": dn_near["near_peak_dist"] if dn_near is not None else np.nan,
        "up_kum": up_near["near_peak_kum"] if up_near is not None else np.nan,
        "dn_kum": dn_near["near_peak_kum"] if dn_near is not None else np.nan,
        "fuel_up_1d": up_fuel1, "fuel_dn_1d": dn_fuel1,
        "fuel_ratio_1d": up_fuel1 / dn_fuel1 if dn_fuel1 else np.nan,
        "rv60": rv60, "fwd_ret60": fwd_ret, "fwd_max_up60": fwd_max_up, "fwd_max_dn60": fwd_max_dn,
        "hour_utc": t0.hour,
    })
events = pd.DataFrame(ev)
events.to_csv(f"{AD}/events.csv", index=False)
print("events rows:", len(events))

# ---------- empirical summaries ----------
print("\n=== 1. Touch rate of nearest strong level by |distance| bucket (window=1 день) ===")
l1 = labels[labels["window"] == "1 день"].copy()
l1["dist_bucket"] = pd.cut(l1["near_peak_dist"].abs(), [0, 50, 100, 200, 400, 800, 1e9])
for H in [60, 240]:
    s = l1[l1["horizon"] == H].groupby("dist_bucket", observed=True)["touched"].agg(["mean", "count"])
    print(f"horizon={H}m:\n{s}")

print("\n=== 2. Fuel asymmetry vs direction of first 60m excursion ===")
e = events.dropna(subset=["fuel_ratio_1d", "fwd_max_up60", "fwd_max_dn60"]).copy()
e["excursion_dir"] = np.where(e["fwd_max_up60"] > e["fwd_max_dn60"], "UP", "DOWN")
e["fuel_bin"] = pd.qcut(e["fuel_ratio_1d"], 4, duplicates="drop")
print(pd.crosstab(e["fuel_bin"], e["excursion_dir"], normalize="index"))

print("\n=== 3. Does price get pulled toward the nearest UP peak? ===")
# distance to up peak should shrink within 60m more often than grow, controlling sign
e2 = events.dropna(subset=["up_dist", "dn_dist", "fwd_ret60"]).copy()
# predicted move direction from magnet pull = sign(up_dist)*-1 weighted vs dn? simple:
e2["pull_up_shrinks"] = np.where(e2["up_dist"] > 0, e2["fwd_ret60"] > 0, e2["fwd_ret60"] < 0)
print("up-peak above mid; ret>0 shrinks dist:", e2["pull_up_shrinks"].mean())

print("\n=== 4. Dead zone (divider lag) vs collisions ===")
if len(coll):
    c = coll.copy()
    c["day"] = c["ts_utc"].dt.date
    print(c.groupby("day")["usd"].agg(["count", "sum", "max"]))
    print("top collisions:")
    print(c.nlargest(10, "usd")[["ts_utc", "window", "side", "level", "usd", "reason"]])

print("\n=== 5. Regime: hourly mid + vol ===")
hh = snap.set_index("ts_utc")["mid"].resample("1h").agg(["first", "max", "min", "last"])
hh["rng"] = hh["max"] - hh["min"]
print(hh.round(1).to_string())
