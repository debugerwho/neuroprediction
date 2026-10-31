#!/usr/bin/env python3
"""Extension: 7d-window targets, reversion-to-origin, by-day regime split."""
import glob
import numpy as np
import pandas as pd

SNAP = pd.read_csv(".cache/parsed/snapshots.csv", parse_dates=["ts_utc"])
LEV = pd.read_csv(".cache/parsed/levels.csv", parse_dates=["ts_utc"])
SNAP["ts_utc"] = pd.to_datetime(SNAP["ts_utc"], utc=True)
LEV["ts_utc"] = pd.to_datetime(LEV["ts_utc"], utc=True)
mid_map = SNAP.set_index("file")["mid"]
LEV["mid"] = LEV["file"].map(mid_map)

cols = ["ts", "open", "high", "low", "close", "vol", "ts_close", "qvol",
        "ntr", "tbase", "tq", "ignore"]
kl = pd.concat([pd.read_csv(f, header=None, names=cols)
                for f in sorted(glob.glob("data/btc_1m/spot/BTCUSDT-1m-*.csv"))],
               ignore_index=True)
kl["ts"] = pd.to_datetime(kl["ts"] // 1000, unit="ms", utc=True)
kl = kl.set_index("ts").sort_index()
kl = kl[~kl.index.duplicated(keep="first")]
minutes = kl.index.values.astype("datetime64[ns]")
close = kl["close"].values
high = kl["high"].values
low = kl["low"].values

out = []
P = out.append
STRONG = 1.0
HOR = 240


def nearest_strong(g, side, mid):
    s = g[(g["side"] == side) & (g["loc"] >= STRONG)]
    if s.empty:
        return None
    if side == "UP":
        s = s[s["level"] > mid]
        return None if s.empty else s.loc[s["level"].idxmin()]
    s = s[s["level"] < mid]
    return None if s.empty else s.loc[s["level"].idxmax()]


def run_window(win):
    lev = LEV[LEV["window"] == win]
    rows = []
    for f, g in lev.groupby("file", sort=False):
        row = SNAP[SNAP["file"] == f].iloc[0]
        t0, mid = row["ts_utc"], row["mid"]
        i0 = int(np.searchsorted(minutes, np.datetime64(t0)))
        if i0 >= len(close) - 1:
            continue
        for side in ("UP", "DOWN"):
            lv = nearest_strong(g, side, mid)
            if lv is None:
                continue
            L = lv["level"]
            up = side == "UP"
            touch_i = None
            for j in range(i0, min(len(close), i0 + HOR + 1)):
                if (high[j] >= L) if up else (low[j] <= L):
                    touch_i = j
                    break
            rec = dict(win=win, file=f, ts=t0, side=side, dist=L - mid,
                       loc=lv["loc"], touched=touch_i is not None)
            if touch_i is not None:
                rec["tt_min"] = touch_i - i0
                c_t = close[touch_i]
                sign = 1 if up else -1
                for h in (15, 30, 60):
                    j = min(len(close) - 1, touch_i + h)
                    rec[f"post{h}"] = (close[j] / c_t - 1) * sign * 1e4
                # reversion to origin: return to pre-touch mid within 60m after touch
                j60 = min(len(close) - 1, touch_i + 60)
                seg = close[touch_i:j60 + 1]
                rec["back_to_mid"] = bool((seg <= mid).any()) if up else bool((seg >= mid).any())
                # touch-bar range
                rec["bar_rng"] = (high[touch_i] - low[touch_i]) / L * 1e4
                rec["day"] = str(t0.date())
            rows.append(rec)
    return pd.DataFrame(rows)


for win in ("1 день", "7 день"):
    D = run_window(win)
    Dt = D[D.touched].copy()
    Dt = Dt.sort_values("ts").drop_duplicates(["side", "day", "lb"] if False else
                                              ["side", Dt["dist"].round().astype(int), "ts"].__class__ and
                                              ["side", (Dt["level"] if "level" in Dt else Dt["dist"]).round(), "ts"], keep="first") \
        if False else Dt.sort_values("ts").drop_duplicates(["side", Dt["dist"].abs().round(), pd.Grouper(key="ts", freq="6h")], keep="first") \
        if False else Dt  # keep all; dedupe below
    # proper dedupe: same side + |dist| bucket 25$ + 6h block
    Dt["blk"] = Dt.ts.dt.floor("6h")
    Dt["dbk"] = (Dt.dist.abs() // 25).astype(int)
    Dt = Dt.sort_values("ts").drop_duplicates(["side", "dbk", "blk"], keep="first")
    P(f"\n##### window={win}: candidates={len(D)} touched={len(D)} deduped_touches={len(Dt)}")
    D["db"] = pd.cut(D.dist.abs(), [0, 100, 200, 400, 800, 1e9])
    D["hit60"] = D.get("tt_min", pd.Series(np.nan, index=D.index)).le(60).fillna(False)
    P("touch@60m by dist:")
    P(D.groupby(["db", "side"], observed=True)["hit60"].agg(["mean", "size"]).round(2).to_string())
    if len(Dt):
        P("post-touch medians by side:")
        P(Dt.groupby("side")[["post15", "post30", "post60"]].median().round(1).to_string())
        P(f"reversion-to-origin within 60m after touch: {Dt.back_to_mid.mean():.2f} "
          f"(UP {Dt[Dt.side=='UP'].back_to_mid.mean():.2f} / DOWN {Dt[Dt.side=='DOWN'].back_to_mid.mean():.2f})")
        P("by day (post30 median / n):")
        P(Dt.groupby("day")[["post30"]].median().round(1).assign(n=Dt.groupby("day").size()).to_string())
        # bar range by side
        P(f"median touch-bar range: {Dt.bar_rng.median():.0f}bp")

# regime conditioning on 1d window: use pre-touch 60m market return sign
D = run_window("1 день")
Dt = D[D.touched].copy()
Dt["blk"] = Dt.ts.dt.floor("6h")
Dt["dbk"] = (Dt.dist.abs() // 25).astype(int)
Dt = Dt.sort_values("ts").drop_duplicates(["side", "dbk", "blk"], keep="first")
i_of = {np.datetime64(t): i for i, t in enumerate(minutes)}
pre = []
for r in Dt.itertuples():
    i0 = int(np.searchsorted(minutes, np.datetime64(r.ts)))
    pre.append((close[min(len(close)-1, i0)] / close[max(0, i0-60)] - 1) * 1e4)
Dt["pre60"] = pre
Dt["regime"] = np.where(Dt.pre60 > 15, "up60", np.where(Dt.pre60 < -15, "dn60", "flat"))
P("\n##### regime (pre60) × side → post30 (through-level bp)")
P(Dt.groupby(["regime", "side"])["post30"].agg(["median", "mean", "size"]).round(1).to_string())
P("\n##### regime × side → reversion-to-origin")
P(Dt.groupby(["regime", "side"])["back_to_mid"].agg(["mean", "size"]).round(2).to_string())

with open(".cache/analysis_react/summary2.txt", "w") as fh:
    fh.write("\n".join(map(str, out)))
print("\n".join(map(str, out)))
