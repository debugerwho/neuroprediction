#!/usr/bin/env python3
"""Reaction-from-levels analysis: how price behaves at/after liquidation-map levels.

Event = first touch of nearest strong 1d/7d level after a snapshot.
Measures approach dynamics, touch-bar shape, post-touch continuation vs rejection,
MFE/MAE, and joins futures metrics (OI, taker flow) at the event minute.
"""
import glob
import numpy as np
import pandas as pd

SNAP = pd.read_csv(".cache/parsed/snapshots.csv", parse_dates=["ts_utc"])
LEV = pd.read_csv(".cache/parsed/levels.csv", parse_dates=["ts_utc"])
SNAP["ts_utc"] = pd.to_datetime(SNAP["ts_utc"], utc=True)
LEV["ts_utc"] = pd.to_datetime(LEV["ts_utc"], utc=True)

# ---------- 1m klines ----------
cols = ["ts", "open", "high", "low", "close", "vol", "ts_close", "qvol",
        "ntr", "tbase", "tq", "ignore"]
kl = []
for f in sorted(glob.glob("data/btc_1m/spot/BTCUSDT-1m-*.csv")):
    d = pd.read_csv(f, header=None, names=cols)
    kl.append(d)
kl = pd.concat(kl, ignore_index=True)
kl["ts"] = pd.to_datetime(kl["ts"] // 1000, unit="ms", utc=True)
kl = kl.set_index("ts").sort_index()
kl = kl[~kl.index.duplicated(keep="first")]
kl["ret1"] = kl["close"].pct_change()
print(f"klines: {len(kl)} bars {kl.index[0]} -> {kl.index[-1]}")

# 5m futures metrics
mfiles = sorted(glob.glob("data/btc_1m/metrics/BTCUSDT-metrics-*.csv"))
if mfiles:
    met = pd.concat([pd.read_csv(f) for f in mfiles], ignore_index=True)
    met["ts"] = pd.to_datetime(met["create_time"], utc=True)
    met = met.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
    met = met.resample("1min").ffill()
else:
    met = None

# ---------- level peaks ----------
LEV["dist"] = LEV["level"] - LEV["ts_utc"].map(lambda t: 0)  # placeholder
# attach mid per snapshot
mid_map = SNAP.set_index("file")["mid"]
LEV["mid"] = LEV["file"].map(mid_map)
LEV["signed_dist"] = LEV["level"] - LEV["mid"]
WIN = "1 день"
lev = LEV[LEV["window"] == WIN].copy()
STRONG = 1.0  # loc threshold


def nearest_strong(g, side, mid):
    s = g[(g["side"] == side) & (g["loc"] >= STRONG)]
    if s.empty:
        return None
    if side == "UP":
        s = s[s["level"] > mid]
        if s.empty:
            return None
        return s.loc[s["level"].idxmin()]
    else:
        s = s[s["level"] < mid]
        if s.empty:
            return None
        return s.loc[s["level"].idxmax()]


events = []
for f, g in lev.groupby("file", sort=False):
    row = SNAP[SNAP["file"] == f].iloc[0]
    t0 = row["ts_utc"]
    mid = row["mid"]
    for side in ("UP", "DOWN"):
        lv = nearest_strong(g, side, mid)
        if lv is None:
            continue
        events.append(dict(file=f, ts=t0, mid=mid, side=side,
                           level=lv["level"], loc=lv["loc"], cum=lv["cum"],
                           dist=lv["level"] - mid))
ev = pd.DataFrame(events)
print(f"candidate targets: {len(ev)} ({(ev.side=='UP').sum()} UP / {(ev.side=='DOWN').sum()} DOWN)")

# ---------- touch + reaction metrics ----------
minutes = kl.index.values.astype("datetime64[ns]")
close = kl["close"].values
high = kl["high"].values
low = kl["low"].values
opens = kl["open"].values


def idx_of(ts):
    return int(np.searchsorted(minutes, np.datetime64(ts)))


HOR = 240
rows = []
for e in ev.itertuples():
    i0 = idx_of(e.ts)
    if i0 >= len(close):
        continue
    L = e.level
    up = e.side == "UP"
    # approach: return over 15m before snapshot
    i_a = max(0, i0 - 15)
    pre15 = close[i0 - 1] / close[i_a] - 1 if i0 > 15 else np.nan
    # find first touch within horizon
    j_end = min(len(close), i0 + HOR + 1)
    touch_i = None
    for j in range(i0, j_end):
        if (high[j] >= L) if up else (low[j] <= L):
            touch_i = j
            break
    touched = touch_i is not None
    rec = dict(file=e.file, ts=e.ts, side=e.side, level=L, dist=e.dist,
               loc=e.loc, cum=e.cum, mid=e.mid, pre15=pre15, touched=touched)
    if touched:
        tt = touch_i - i0
        rec["tt_min"] = tt
        # touch bar shape
        if up:
            rec["wick_beyond"] = (high[touch_i] - L) / L * 1e4
            rec["close_rel"] = (close[touch_i] - L) / L * 1e4  # >0 close above level
        else:
            rec["wick_beyond"] = (L - low[touch_i]) / L * 1e4
            rec["close_rel"] = (L - close[touch_i]) / L * 1e4
        # tape: taker buy share & volume surge around touch
        vol = kl["vol"].values
        tbase = kl["tbase"].values
        j_lo = max(0, touch_i - 14)
        v_now = vol[j_lo:touch_i + 1].sum()
        v_base = vol[max(0, j_lo - 60):j_lo].sum() / 4.0 + 1e-9
        rec["vol_surge"] = v_now / v_base
        rec["taker_share"] = tbase[j_lo:touch_i + 1].sum() / max(vol[j_lo:touch_i + 1].sum(), 1e-9)
        # post-touch returns from touch bar close
        c_t = close[touch_i]
        sign = 1 if up else -1  # "toward level direction" positive
        for h in (5, 15, 30, 60):
            j = min(len(close) - 1, touch_i + h)
            # move in the direction of the sweep (through the level)
            rec[f"post{h}"] = (close[j] / c_t - 1) * sign * 1e4
        # MFE/MAE over 30m after touch, direction-of-sweep positive
        j30 = min(len(close), touch_i + 31)
        seg_h = high[touch_i:j30]
        seg_l = low[touch_i:j30]
        if up:
            rec["mfe"] = (seg_h.max() / c_t - 1) * 1e4
            rec["mae"] = (c_t / seg_l.min() - 1) * 1e4  # adverse = drop
        else:
            rec["mfe"] = (c_t / seg_l.min() - 1) * 1e4
            rec["mae"] = (seg_h.max() / c_t - 1) * 1e4
        # rejection: within 15m after touch, price closes back near side
        j15 = min(len(close) - 1, touch_i + 15)
        if up:
            rec["rejected"] = close[j15] < L
        else:
            rec["rejected"] = close[j15] > L
        # metrics at touch minute
        if met is not None and len(met):
            t_touch = pd.Timestamp(minutes[touch_i]).tz_localize("UTC")
            try:
                m = met.loc[met.index <= t_touch].iloc[-1]
                rec["oi"] = m["sum_open_interest"]
                rec["taker_ls"] = m["sum_taker_long_short_vol_ratio"]
                rec["ls_ratio"] = m["count_long_short_ratio"]
                # OI change over last 15 min
                past = met.loc[met.index <= t_touch - pd.Timedelta(minutes=15)]
                if len(past):
                    rec["oi_d15"] = m["sum_open_interest"] - past.iloc[-1]["sum_open_interest"]
            except Exception:
                pass
    rows.append(rec)

D = pd.DataFrame(rows)
D.to_csv(".cache/analysis_react/events.csv", index=False)
print(f"events with touch info: {D.touched.sum()} / {len(D)}")

# ---------- dedupe touches: same level bucket & day & side, keep first ----------
Dt = D[D.touched].copy()
Dt["day"] = Dt.ts.dt.floor("D")
Dt["lbucket"] = (Dt.level // 25).astype(int)
Dt = Dt.sort_values("ts").drop_duplicates(["side", "lbucket", "day"], keep="first")
print(f"deduped touch events: {len(Dt)}")

out = []
P = out.append

P("=== SAMPLE ===")
P(f"snapshots={SNAP.shape[0]} span {SNAP.ts_utc.min()} -> {SNAP.ts_utc.max()}")
P(f"touch rate 60m: {D[D.tt_min<=60].shape[0] if 'tt_min' in D else 0}/n; 240m touches={Dt.shape[0]} deduped")

# 1. touch rate by distance (60m)
D60 = D.copy()
D60["hit60"] = D60.get("tt_min", pd.Series(np.nan, index=D60.index)).le(60).fillna(False)
D60.loc[~D60.touched, "hit60"] = False
bins = [0, 100, 200, 400, 800, 1e9]
D60["db"] = pd.cut(D60.dist.abs(), bins)
P("\n=== A. P(touch level | distance), window=1d ===")
P(D60.groupby(["db", "side"], observed=True)["hit60"].agg(["mean", "size"]).round(3).to_string())

# 2. reaction AFTER touch: continuation vs rejection
Dt["strength"] = pd.cut(Dt["loc"], [1, 2, 4, 8, 1e9])
P("\n=== B. Post-touch behaviour (deduped; positive = move THROUGH the level) ===")
P(f"all medians: {Dt[['post5','post15','post30','post60','mfe','mae']].median().round(1).to_dict()}")
for side in ("UP", "DOWN"):
    s = Dt[Dt.side == side]
    P(f"{side}: n={len(s)} median post5/15/30/60={s.post5.median():.0f}/{s.post15.median():.0f}/"
      f"{s.post30.median():.0f}/{s.post60.median():.0f}bp  MFE={s.mfe.median():.0f} MAE={s.mae.median():.0f} "
      f"reject15={s.rejected.mean():.2f}")

P("\n=== C. By strength of touched level (loc) ===")
P(Dt.groupby("strength", observed=True)[["post15", "post30", "mfe", "mae", "rejected"]]
  .agg(["median", "mean", "size"]).round(2).to_string())

P("\n=== D. By approach speed (pre15 return toward the level, bp) ===")
Dt["approach"] = Dt["pre15"] * (1 if True else 1)
Dt["appr_toward"] = np.where(Dt.side == "UP", Dt.pre15, -Dt.pre15) * 1e4
Dt["app_bin"] = pd.cut(Dt.appr_toward, [-1e9, 0, 30, 80, 1e9])
P(Dt.groupby("app_bin", observed=True)[["post15", "post30", "rejected", "tt_min"]]
  .agg(["median", "mean", "size"]).round(2).to_string())

P("\n=== E. Touch-bar shape ===")
P(f"median wick_beyond={Dt.wick_beyond.median():.1f}bp, close_rel={Dt.close_rel.median():.1f}bp; "
  f"P(close beyond level at touch bar)={(Dt.close_rel>0).mean():.2f}")
P(f"P(wick>50bp)={ (Dt.wick_beyond>50).mean():.2f} → median post15 when big wick: "
  f"{Dt.loc[Dt.wick_beyond>50,'post15'].median():.0f}bp vs small wick "
  f"{Dt.loc[Dt.wick_beyond<=50,'post15'].median():.0f}bp")

# 5. side asymmetry: what happens AFTER an UP sweep → drift down?
P("\n=== F. Cross-side: after UP touch, DOWN-side movement next 30m (signed close-to-close, bp) ===")
# reuse close series: post30 already signed through UP level (= raw return*1e4 for UP)
# For "reversal" measure: raw return after UP sweep
for side in ("UP", "DOWN"):
    s = Dt[Dt.side == side]
    raw = s.post30 * np.where(s.side == "UP", 1, -1)  # raw bp return
    P(f"{side} touches: raw 30m return median={raw.median():.1f}bp mean={raw.mean():.1f}bp "
      f"P(>0)={(raw>0).mean():.2f}")

# 6. net edge MFE-MAE
Dt["edge30"] = Dt.mfe - Dt.mae
P("\n=== G. Net edge over 30m after touch (MFE-MAE, bp) ===")
P(Dt.groupby("side")["edge30"].agg(["median", "mean", "size"]).round(1).to_string())
P(Dt.groupby(pd.cut(Dt.dist, bins), observed=True)["edge30"].agg(["median", "mean", "size"]).round(1).to_string())

# 7. futures metrics at touch
if "oi_d15" in Dt.columns:
    P("\n=== H. Futures metrics at touch minute (median) ===")
    P(Dt.groupby("side")[["oi_d15", "taker_ls", "ls_ratio"]].median().round(4).to_string())
    # does taker flow direction predict continuation?
    Dt["taker_up"] = np.where(Dt.side == "UP", Dt.taker_ls - 1, 1 - 1/Dt.taker_ls.clip(lower=0.01))
    Dt["cont30"] = (Dt.post30 > 0).astype(int)
    q = Dt.dropna(subset=["taker_ls", "post30"])
    if len(q) > 30:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(q.cont30, q.taker_ls * np.where(q.side == "UP", 1, -1))
        P(f"AUC(taker_ls signed→continuation30)={auc:.3f} n={len(q)}")
        auc_oi = roc_auc_score(q.cont30, q.oi_d15.fillna(0) * np.where(q.side == "UP", -1, 1))
        P(f"AUC(-oi_d15 signed→continuation30)={auc_oi:.3f}  (OI rise vs sweep direction)")

# 8. logistic on rejection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
q = Dt.dropna(subset=["post15", "pre15", "taker_ls", "oi_d15"]).copy()
if len(q) > 60:
    q["y"] = (q.post15 < 0).astype(int)  # reversed against sweep direction
    X = q[["dist", "loc", "appr_toward", "wick_beyond", "close_rel", "taker_ls", "oi_d15"]].fillna(0)
    Xtr, Xte, ytr, yte = train_test_split(X, q.y, test_size=0.3, random_state=0, stratify=q.y)
    lr = LogisticRegression(max_iter=2000).fit((Xtr - Xtr.mean()) / Xtr.std(), ytr)
    auc = roc_auc_score(yte, lr.predict_proba((Xte - Xtr.mean()) / Xtr.std())[:, 1])
    coefs = pd.Series(lr.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
    P(f"\n=== I. Reversal-in-15m model: AUC={auc:.3f} base={q.y.mean():.2f} n={len(q)} ===")
    P(coefs.round(3).to_string())

# 9. big-cluster sweep aftermath (loc top quartile)
thr = Dt["loc"].quantile(0.75)
big = Dt[Dt["loc"] >= thr]
P(f"\n=== J. Strongest sweeps (loc>={thr:.2f}, n={len(big)}) ===")
P(f"median post5/30/60={big.post5.median():.0f}/{big.post30.median():.0f}/{big.post60.median():.0f}bp; "
  f"reject15={big.rejected.mean():.2f}; tt_min median={big.tt_min.median():.0f}")

# 10. controls: non-touched targets drift
ctrl = D[~D.touched].copy()
P(f"\n=== K. Controls (targets never touched in 240m, n={len(ctrl)}) ===")
P(f"median |dist| at snapshot={ctrl.dist.median():.0f}bp-ish $={ctrl.dist.abs().median():.0f}")

# L. tape confirmation
if "taker_share" in Dt.columns:
    P("\n=== L. Tape at touch (15m window ending at touch bar) ===")
    P(Dt.groupby("side")[["vol_surge", "taker_share"]].median().round(3).to_string())
    q2 = Dt.dropna(subset=["taker_share", "post30"]).copy()
    if len(q2) > 40:
        from sklearn.metrics import roc_auc_score as _auc
        P(f"AUC(taker_share signed → post30 through-level)="
          f"{_auc((q2.post30>0).astype(int), q2.taker_share*np.where(q2.side=='UP',1,-1)):.3f} n={len(q2)}")
        P(f"AUC(vol_surge → |post30|>10bp)="
          f"{_auc((q2.post30.abs()>10).astype(int), q2.vol_surge):.3f}")

# M. loc bimodality: moderate vs monster
med = Dt["loc"].median()
P(f"\n=== M. Bimodality by loc (median split at {med:.2f}) ===")
P(Dt.groupby(Dt["loc"] >= med)[["post15", "post30", "post60", "rejected", "tt_min"]]
  .agg(["median", "mean", "size"]).round(2).to_string())

# N. collisions aftermath (dead-zone sweeps)
try:
    COLL = pd.read_csv(".cache/parsed/collisions.csv", parse_dates=["ts_utc"])
    COLL["ts_utc"] = pd.to_datetime(COLL["ts_utc"], utc=True)
    P("\n=== N. Collision (dead-zone sweep) aftermath ===")
    for f, g in COLL.groupby("file"):
        t0 = g["ts_utc"].iloc[0]
        i0 = int(np.searchsorted(minutes, np.datetime64(t0)))
        if i0 >= len(close) - 180:
            P(f"{f}: skip (edge)")
            continue
        c0 = close[i0]
        usd = g["usd"].sum()
        r = lambda h: (close[min(len(close)-1, i0 + h)] / c0 - 1) * 1e4
        sides = ",".join(sorted(g["side"].unique()))
        P(f"{f} t={t0} ${usd/1e6:.0f}M sides={sides} ret+30/60/120/180m = "
          f"{r(30):.0f}/{r(60):.0f}/{r(120):.0f}/{r(180):.0f} bp")
except Exception as ex:
    P(f"collisions block failed: {ex}")

with open(".cache/analysis_react/summary.txt", "w") as fh:
    fh.write("\n".join(map(str, out)))
print("\n".join(map(str, out)))
