#!/usr/bin/env python3
"""Magnet-competition 'race' analysis + simple ML baseline."""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

AD = ".cache/analysis"
k_frames = []
cols = ["ot", "open", "high", "low", "close", "volume", "ct", "qv", "tr", "tb", "tq", "ig"]
for f in sorted(os.listdir("data/btc_1m/spot")):
    df = pd.read_csv(f"data/btc_1m/spot/{f}", header=None, names=cols)
    k_frames.append(df)
k = pd.concat(k_frames).drop_duplicates("ot").sort_values("ot")
k["ts"] = pd.to_datetime(k["ot"] // 1000, unit="ms", utc=True)

tg = pd.read_csv(f"{AD}/targets.csv")
tg["ts_utc"] = pd.to_datetime(tg["ts_utc"], utc=True)
ev = pd.read_csv(f"{AD}/events.csv")
ev["ts_utc"] = pd.to_datetime(ev["ts_utc"], utc=True)

# --- race: for each snapshot, nearest UP peak vs nearest DOWN peak (1d window) ---
w = tg[tg["window"] == "1 день"].pivot_table(
    index="file", columns="side", values=["near_peak_level", "near_peak_kum", "near_peak_dist"]
)
w.columns = [f"{a}_{b}" for a, b in w.columns]
w = w.reset_index().merge(ev[["file", "ts_utc", "mid"]], on="file")

def first_hit(t0, lvl, H=60):
    m = k[(k["ts"] > t0) & (k["ts"] <= t0 + pd.Timedelta(minutes=H))]
    hit = m[m["high"] >= lvl]
    return (hit["ts"].iloc[0] - t0).total_seconds() / 60 if len(hit) else np.nan

def first_hit_dn(t0, lvl, H=60):
    m = k[(k["ts"] > t0) & (k["ts"] <= t0 + pd.Timedelta(minutes=H))]
    hit = m[m["low"] <= lvl]
    return (hit["ts"].iloc[0] - t0).total_seconds() / 60 if len(hit) else np.nan

rows = []
for r in w.dropna(subset=["near_peak_level_UP", "near_peak_level_DOWN"]).itertuples():
    tu = first_hit(r.ts_utc, r.near_peak_level_UP)
    td = first_hit_dn(r.ts_utc, r.near_peak_level_DOWN)
    winner = None
    if not np.isnan(tu) and (np.isnan(td) or tu < td):
        winner = "UP"
    elif not np.isnan(td) and (np.isnan(tu) or td < tu):
        winner = "DOWN"
    rows.append({
        "file": r.file, "up_dist": r.near_peak_dist_UP, "dn_dist": -r.near_peak_dist_DOWN,
        "up_kum": r.near_peak_kum_UP, "dn_kum": r.near_peak_kum_DOWN,
        "dist_gap": abs(r.near_peak_dist_UP) - abs(r.near_peak_dist_DOWN),
        "kum_ratio": r.near_peak_kum_UP / r.near_peak_kum_DOWN if r.near_peak_kum_DOWN else np.nan,
        "winner": winner,
    })
race = pd.DataFrame(rows)
print("=== RACE (1d window, 60m horizon) ===")
print("n=", len(race), "winner counts:\n", race["winner"].value_counts(dropna=False))
r2 = race.dropna(subset=["winner"]).copy()
r2["closer_side"] = np.where(r2["dist_gap"] < 0, "UP", "DOWN")
acc = (r2["closer_side"] == r2["winner"]).mean()
print(f"closer-side predicts winner: {acc:.3f} (base {r2['winner'].value_counts(normalize=True).max():.3f})")
r2["stronger_side"] = np.where(r2["kum_ratio"] > 1, "UP", "DOWN")
acc2 = (r2["stronger_side"] == r2["winner"]).mean()
print(f"stronger-кум predicts winner: {acc2:.3f}")

# logistic baseline on race features
X = r2[["up_dist", "dn_dist", "up_kum", "dn_kum", "dist_gap", "kum_ratio"]].replace([np.inf, -np.inf], np.nan).fillna(0)
y = (r2["winner"] == "UP").astype(int)
split = int(len(r2) * 0.6)
# time order: race rows follow snapshot order already (w from pivot on file order = chronological? pivot sorts by file name = chronological ✓)
Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
m = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
p = m.predict_proba(Xte)[:, 1]
print(f"logreg test AUC={roc_auc_score(yte, p):.3f} n_test={len(yte)}")
base = yte.mean()
pred = (p > 0.5).astype(int)
print(f"logreg test acc={(pred == yte.values).mean():.3f} base={max(base, 1-base):.3f}")

# --- touch model on events (60m, up-peak touched) ---
lab = pd.read_csv(f"{AD}/labels.csv")
lab["ts_utc"] = pd.to_datetime(lab["ts_utc"], utc=True)
l60 = lab[(lab["horizon"] == 60) & (lab["window"] == "1 день")]
mrg = l60.merge(ev[["file", "mid", "spread", "book_imb", "dead_up", "dead_dn", "up_dist", "dn_dist", "up_kum", "dn_kum", "fuel_ratio_1d", "rv60", "hour_utc"]], on="file")
mrg["signed_dist"] = mrg["near_peak_dist"]
feats = ["signed_dist", "near_peak_kum", "book_imb", "dead_up", "dead_dn", "up_dist", "dn_dist", "up_kum", "dn_kum", "fuel_ratio_1d", "rv60", "hour_utc", "spread"]
X = mrg[feats].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())
y = mrg["touched"]
split = int(len(mrg) * 0.6)
m = LogisticRegression(max_iter=2000).fit(X.iloc[:split], y.iloc[:split])
p = m.predict_proba(X.iloc[split:])[:, 1]
print(f"\ntouch-60m model: AUC={roc_auc_score(y.iloc[split:], p):.3f} n={len(mrg)} base={y.iloc[split:].mean():.3f}")
imp = pd.Series(np.abs(m.coef_[0]), index=feats).sort_values(ascending=False)
print("top coefficients:\n", imp.head(6))
