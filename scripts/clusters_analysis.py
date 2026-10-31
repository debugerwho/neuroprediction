#!/usr/bin/env python3
"""Full-profile cluster analysis: build liquidation clusters (loc>=1 runs),
detect walls (loc jumps, cum drops), then follow the 1m price path and classify
what price did with EACH cluster: reached / entered / pierced / retracted,
local trend flip at touch, and cluster-to-cluster consumption diary."""
import glob
import numpy as np
import pandas as pd

SNAP = pd.read_csv(".cache/parsed/snapshots.csv", parse_dates=["ts_utc"])
LEV = pd.read_csv(".cache/parsed/levels.csv", parse_dates=["ts_utc"])
SNAP["ts_utc"] = pd.to_datetime(SNAP["ts_utc"], utc=True)
LEV["ts_utc"] = pd.to_datetime(LEV["ts_utc"], utc=True)
SNAP = SNAP.sort_values("ts_utc").reset_index(drop=True)

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

def idx_of(ts):
    return int(np.searchsorted(minutes, np.datetime64(ts)))

WIN = "1 день"
GAP = 100.0        # max $ between adjacent levels of one cluster
LOC_MIN = 1.0      # elevated loc threshold
SIG_MIN = 2.0      # significant cluster: max loc
HOR = 4 * 60       # forward horizon minutes
WALL_RATIO = 5.0   # sharp loc jump vs previous (closer) cluster

lev = LEV[LEV["window"] == WIN].copy()
lev["mid"] = lev["file"].map(SNAP.set_index("file")["mid"])

def build_clusters(g, side, mid):
    s = g[g["side"] == side]
    s = s[s["level"] > mid] if side == "UP" else s[s["level"] < mid]
    if s.empty:
        return []
    s = s.sort_values("level").reset_index(drop=True)
    clusters, cur = [], []
    for i, r in s.iterrows():
        if r["loc"] < LOC_MIN:
            if cur:
                clusters.append(cur)
                cur = []
            continue
        if cur and (r["level"] - cur[-1]["level"]) > GAP:
            clusters.append(cur)
            cur = []
        cur.append(r)
    if cur:
        clusters.append(cur)
    out = []
    for c in clusters:
        df = pd.DataFrame(c)
        max_loc = df["loc"].max()
        if max_loc < SIG_MIN:
            continue
        center = float(df.loc[df["loc"].idxmax(), "level"])
        cum_series = df["cum"].values
        cum_drop = bool(np.any(np.diff(cum_series) < -1e-6))
        out.append(dict(side=side, lo=float(df["level"].min()), hi=float(df["level"].max()),
                        center=center, max_loc=float(max_loc), sum_loc=float(df["loc"].sum()),
                        nlev=len(df), cum_drop=cum_drop,
                        cum_end=float(cum_series[-1]), dist=center - mid))
    # sort by distance from mid, tag wall vs previous closer cluster
    out.sort(key=lambda d: abs(d["dist"]))
    prev = None
    for d in out:
        d["wall"] = False
        if prev is not None:
            d["wall_ratio"] = d["max_loc"] / max(prev["max_loc"], 0.2)
            if d["wall_ratio"] >= WALL_RATIO:
                d["wall"] = True
        prev = d
    return out

# ---------- build all clusters ----------
rows = []
for f, g in lev.groupby("file", sort=False):
    row = SNAP[SNAP["file"] == f].iloc[0]
    t0, mid = row["ts_utc"], row["mid"]
    i0 = idx_of(t0)
    if i0 >= len(close) - 30:
        continue
    for side in ("UP", "DOWN"):
        for c in build_clusters(g, side, mid):
            c.update(file=f, ts=t0, mid=mid, i0=i0)
            rows.append(c)
CL = pd.DataFrame(rows)
print(f"clusters: {len(CL)} from {CL.file.nunique()} snapshots")

# ---------- follow price path for each cluster ----------
res = []
for c in CL.itertuples():
    i0, up = c.i0, c.side == "UP"
    jend = min(len(close), i0 + HOR + 1)
    if up:
        span_lo, span_hi, center = c.lo, c.hi, c.center
        # touch span: high >= lo; enter: high >= center; pierce: high >= hi
        t_span = t_enter = t_pierce = None
        for j in range(i0, jend):
            if t_span is None and high[j] >= span_lo:
                t_span = j
            if t_enter is None and high[j] >= center:
                t_enter = j
            if t_pierce is None and high[j] >= span_hi:
                t_pierce = j
                break
            if t_pierce is None and j > i0 + 240 and t_enter is not None:
                break  # keep scanning to pierce within horizon but cap lazily
    else:
        span_lo, span_hi, center = c.lo, c.hi, c.center
        t_span = t_enter = t_pierce = None
        for j in range(i0, jend):
            if t_span is None and low[j] <= span_hi:
                t_span = j
            if t_enter is None and low[j] <= center:
                t_enter = j
            if t_pierce is None and low[j] <= span_lo:
                t_pierce = j
                break
            if t_pierce is None and j > i0 + 240 and t_enter is not None:
                break
    rec = dict(ts=c.ts, side=c.side, center=c.center, lo=c.lo, hi=c.hi,
               max_loc=c.max_loc, sum_loc=c.sum_loc, nlev=c.nlev, dist=c.dist,
               wall=c.wall, cum_drop=c.cum_drop,
               t_span=t_span is not None, t_enter=t_enter is not None,
               t_pierce=t_pierce is not None,
               tt=(t_span - i0) if t_span is not None else np.nan)
    sign = 1 if up else -1
    if t_enter is not None:
        c_e = close[t_enter]
        for h in (15, 30, 60):
            j = min(len(close) - 1, t_enter + h)
            rec[f"post{h}"] = (close[j] / c_e - 1) * sign * 1e4
        # trend before/after touch of span (15m slope)
        i_t = t_span if t_span is not None else t_enter
        pre = (close[i_t] / close[max(0, i_t - 15)] - 1) * sign * 1e4
        post = (close[min(len(close)-1, i_t + 15)] / close[i_t] - 1) * sign * 1e4
        rec["pre15"] = pre
        rec["post15"] = post
        rec["trend_flip"] = bool(pre * post < 0 and abs(post) > 2)
        # retraction after enter: close back beyond near edge within 60m
        seg = close[t_enter:min(len(close), t_enter + 61)]
        if up:
            rec["retract_full"] = bool((seg <= span_lo).any())
            rec["retract_center"] = bool((seg <= center).any())
        else:
            rec["retract_full"] = bool((seg >= span_hi).any())
            rec["retract_center"] = bool((seg >= center).any())
    res.append(rec)

R = pd.DataFrame(res)
R.to_csv(".cache/analysis_react/clusters_path.csv", index=False)

# baseline trend-flip rate at random minutes
rng = np.random.default_rng(1)
flips = []
for _ in range(4000):
    i = rng.integers(15, len(close) - 16)
    pre = close[i] / close[i - 15] - 1
    post = close[i + 15] / close[i] - 1
    flips.append(pre * post < 0 and abs(post * 1e4) > 2)
base_flip = np.mean(flips)

out = []
P = out.append

P("=== 1. Сколько кластеров и какие они ===")
P(f"кластеров всего: {len(R)} (UP { (R.side=='UP').sum() } / DOWN { (R.side=='DOWN').sum() })")
P(f"расстояние до центра: медиана ${R.dist.abs().median():.0f}, p75 ${R.dist.abs().quantile(0.75):.0f}, макс ${R.dist.abs().max():.0f}")
P(f"сила (max loc): медиана {R.max_loc.median():.1f}, p90 {R.max_loc.quantile(0.9):.1f}, макс {R.max_loc.max():.0f}")
P(f"стен (скачок лок к предыдущему кластеру >= {WALL_RATIO}x): {R.wall.sum()} ({R.wall.mean()*100:.0f}%)")
P(f"кластеров с падением кумулятивного внутри: {R.cum_drop.sum()} ({R.cum_drop.mean()*100:.0f}%)")

P("\n=== 2. Судьба кластеров (путь 4 часа от снимка) ===")
P(f"дошли до края (t_span): {R.t_span.mean()*100:.0f}%")
P(f"вошли внутрь, до центра (t_enter): {R.t_enter.mean()*100:.0f}%")
P(f"прошили насквозь весь кластер (t_pierce): {R.t_pierce.mean()*100:.0f}%")
R["sbin"] = pd.cut(R.max_loc, [0, 2, 4, 8, 16, 1e9], labels=["2-4", "4-8", "8-16", "16-32", "32+"])
entered = R[R.t_enter]
P(f"среди вошедших: прошили насквозь {entered.t_pierce.mean()*100:.0f}%, "
  f"полностью откатились обратно за 60 мин {entered.retract_full.mean()*100:.0f}%, "
  f"хотя бы вернулись к центру {entered.retract_center.mean()*100:.0f}%")

P("\n=== 3. Зависимость от силы кластера ===")
P(R.groupby("sbin", observed=True)[["t_span", "t_enter", "t_pierce"]].mean().round(2).to_string())
P("среди вошедших, по силе: откат к центру / насквозь / смена тренда:")
P(entered.groupby("sbin", observed=True)[["retract_center", "trend_flip", "post30"]].agg(
    ["mean", "size"]).round(2).to_string())

P("\n=== 4. Реакция в момент входа в центр ===")
P(f"медиана post15/30/60 сквозь центр: {entered.post15.median():.0f}/{entered.post30.median():.0f}/{entered.post60.median():.0f} bp")
P(f"смена локального тренда (знак 15-мин наклона до/после) при входе: {entered.trend_flip.mean()*100:.0f}% "
  f"(на произвольных минутах: {base_flip*100:.0f}%)")
P(f"подход к кластеру: медианный 15-мин наклон до касания {entered.pre15.median():.1f} bp; "
  f"когда до касания шли ОТ кластера (pre15<0): пост-15 {entered.loc[entered.pre15<0,'post15'].median():.1f} bp (n={(entered.pre15<0).sum()}); "
  f"когда шли К нему (pre15>0): {entered.loc[entered.pre15>0,'post15'].median():.1f} bp (n={(entered.pre15>0).sum()})")

P("\n=== 5. СТЕНЫ: места где лок падает резко (wall_ratio>=5) или кумулятивный уменьшается ===")
walls = R[R.wall]
cdrops = R[R.cum_drop]
P(f"стены (скачок лок вверх при движении от цены): n={len(walls)}")
if len(walls):
    P(f"  до стен дошли/вошли/прошили: {walls.t_span.mean()*100:.0f}%/{walls.t_enter.mean()*100:.0f}%/{walls.t_pierce.mean()*100:.0f}%")
    we = walls[walls.t_enter]
    if len(we):
        P(f"  медиана post30 после входа в за стеной: {we.post30.median():.0f} bp; откат к центру {we.retract_center.mean()*100:.0f}%")
P(f"кластеры с падением кумулятивного: n={len(cdrops)}")
if len(cdrops):
    P(f"  дошли/вошли/прошили: {cdrops.t_span.mean()*100:.0f}%/{cdrops.t_enter.mean()*100:.0f}%/{cdrops.t_pierce.mean()*100:.0f}%")
    ce = cdrops[cdrops.t_enter]
    if len(ce):
        P(f"  медиана post30: {ce.post30.median():.0f} bp; откат к центру {ce.retract_center.mean()*100:.0f}%; "
          f"смена тренда {ce.trend_flip.mean()*100:.0f}%")

P("\n=== 6. Куда шла цена между снапшотами (направление к ближайшему кластеру) ===")
pull = []
files = SNAP.sort_values("ts_utc").to_dict("records")
for a, b in zip(files[:-1], files[1:]):
    ia, ib = idx_of(a["ts_utc"]), idx_of(b["ts_utc"])
    if ib <= ia or ib - ia > 90:
        continue
    dprice = close[ib] - close[ia]
    if abs(dprice) < 5:
        continue
    # distance to nearest cluster on the side of the move, at snapshot a
    sub = CL[CL.file == a["file"]]
    side = "UP" if dprice > 0 else "DOWN"
    cand = sub[sub.side == side]
    if cand.empty:
        continue
    near = cand.iloc[int(np.argmin(np.abs(cand.dist.values)))]
    d0 = abs(near["center"] - close[ia])
    d1 = abs(near["center"] - close[ib])
    pull.append(d1 < d0)
P(f"цена в {np.mean(pull)*100:.0f}% случаях между снапшотами сокращала расстояние до ближайшего кластера в свою сторону (n={len(pull)})")

P("\n=== 7. Дневник потребления: какие кластеры съедены (дедуп по дню/стороне/25$) ===")
cons = R[R.t_enter].copy()
cons["day"] = cons.ts.dt.floor("D")
cons["bk"] = (cons.center // 25).astype(int)
cons = cons.sort_values("ts").drop_duplicates(["side", "bk", "day"], keep="first")
cons = cons.sort_values("ts")
for r in cons.itertuples():
    pierce = "прошили" if r.t_pierce else ("вошли" if r.t_span else "?")
    ret = f"пост30={r.post30:+.0f}bp" if not np.isnan(r.post30) else ""
    P(f"{r.ts} {r.side:4s} центр={r.center:.0f} сила={r.max_loc:.0f} nlev={r.nlev} "
      f"dist@снимок={r.dist:+.0f} {pierce} {ret} откат={getattr(r,'retract_center','-')}")

with open(".cache/analysis_react/clusters_summary.txt", "w") as fh:
    fh.write("\n".join(map(str, out)))
print("\n".join(map(str, out)))
