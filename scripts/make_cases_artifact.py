#!/usr/bin/env python3
"""Build an artifact with EVERY case of price entering a liquidation cluster:
per-case chart (price path + cluster band + event marks), compact timeline,
plus wall-bounce cases and the 29.10 cascade chain. Outputs artifacts/behavior_cases.html + .md + .csv"""
import base64
import glob
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

OUT = "artifacts"
IMG = os.path.join(OUT, "img")
os.makedirs(IMG, exist_ok=True)

SNAP = pd.read_csv(".cache/parsed/snapshots.csv", parse_dates=["ts_utc"])
SNAP["ts_utc"] = pd.to_datetime(SNAP["ts_utc"], utc=True)
R = pd.read_csv(".cache/analysis_react/clusters_path.csv")
R["ts"] = pd.to_datetime(R["ts"], utc=True)

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

def fmt_t(ts):
    return pd.Timestamp(ts).strftime("%d.%m %H:%M")

def case_events(row):
    """Recompute precise event minutes for one cluster row."""
    i0 = idx_of(row.ts)
    up = row.side == "UP"
    lo, hi, c = row.lo, row.hi, row.center
    jend = min(len(close), i0 + 4 * 60 + 1)
    t_edge = t_center = t_pierce = None
    for j in range(i0, jend):
        if t_edge is None and ((high[j] >= lo) if up else (low[j] <= hi)):
            t_edge = j
        if t_center is None and ((high[j] >= c) if up else (low[j] <= c)):
            t_center = j
        if t_pierce is None and ((high[j] >= hi) if up else (low[j] <= lo)):
            t_pierce = j
            break
    ev = dict(t_edge=t_edge, t_center=t_center, t_pierce=t_pierce)
    if t_center is not None:
        seg = slice(t_center, min(len(close), t_center + 61))
        if up:
            ret = np.where(close[seg] <= c)[0]
            exc_i = t_center + int(np.argmax(high[seg]))
            ev["exc"] = (high[exc_i] - c)
            ev["exc_t"] = exc_i
        else:
            ret = np.where(close[seg] >= c)[0]
            exc_i = t_center + int(np.argmin(low[seg]))
            ev["exc"] = (c - low[exc_i])
            ev["exc_t"] = exc_i
        ev["ret_center"] = t_center + int(ret[0]) if len(ret) else None
        for h in (15, 30, 60):
            j = min(len(close) - 1, t_center + h)
            ev[f"p{h}"] = (close[j] / c - 1) * (1 if up else -1) * 1e4
    if t_edge is not None:
        seg2 = slice(t_edge, min(len(close), t_edge + 61))
        if up:
            pen = np.where(high[seg2] >= lo)[0]
            deep_i = t_edge + int(np.argmax(high[seg2]))
            ev["deep"] = high[deep_i] - lo
            ev["deep_t"] = deep_i
        else:
            deep_i = t_edge + int(np.argmin(low[seg2]))
            ev["deep"] = lo - low[deep_i]  # span_hi side: penetration = span_hi - min low? use hi
            ev["deep"] = hi - low[deep_i] if (hi - low[deep_i]) > 0 else 0
            ev["deep_t"] = deep_i
    return ev

# ---------------- unique ENTERED cases ----------------
ent = R[R.t_enter == True].copy()
ent["key"] = ent.side + "|" + (ent.center // 25 * 25).astype(int).astype(str) + "|" + ent.ts.dt.floor("D").astype(str)
ent = ent.sort_values("ts")
dup_map = ent.groupby("key").size()
cases = ent.drop_duplicates("key", keep="first").copy()
print(f"entered rows={len(ent)} unique events={len(cases)}")

rows_out = []
for i, (idx, r) in enumerate(cases.iterrows(), 1):
    ev = case_events(r)
    pierced = ev["t_pierce"] is not None
    ret = ev["ret_center"] is not None
    if pierced and ret:
        typ = "прошил_но_вернулся"
    elif pierced and not ret:
        typ = "прошил_и_ушёл"
    elif (not pierced) and ret:
        typ = "зашёл_и_вернулся"
    else:
        typ = "зашёл_и_задержался"
    r2 = r.to_dict()
    r2.update(ev, typ=typ, case_id=i, dups=int(dup_map[r["key"]]))
    rows_out.append(r2)

# ---------------- wall edge-bounce cases (never entered) ----------------
eb = R[(R.wall == True) & (R.t_span == True) & (R.t_enter == False)].copy()
eb["key"] = eb.side + "|" + (eb.center // 25 * 25).astype(int).astype(str) + "|" + eb.ts.dt.floor("D").astype(str)
eb = eb.sort_values("ts").drop_duplicates("key", keep="first")
print(f"wall edge-bounce cases: {len(eb)}")

# ---------------- charts ----------------
def make_chart(r, ev, path, label_extra=""):
    up = r.side == "UP"
    i_c = ev.get("t_center") if ev.get("t_center") is not None else ev.get("t_edge")
    if i_c is None:
        i_c = idx_of(r.ts)
    firsts = [x for x in (ev.get("t_edge"), ev.get("t_center"), i_c) if x is not None]
    a, b = max(0, min(firsts) - 25), min(len(close), i_c + 121)
    t = minutes[a:b]
    px = close[a:b]
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=90)
    ax.plot(t, px, lw=1.2, color="#1f77b4")
    ax.axhspan(r.lo, r.hi, color=("#c62828" if up else "#2e7d32"), alpha=0.12)
    ax.axhline(r.center, color=("#c62828" if up else "#2e7d32"), lw=1, ls="--", alpha=0.8)
    marks = [("касание края", ev.get("t_edge"), "#555"),
             ("вход в центр", ev.get("t_center"), "#d81b60"),
             ("прошил насквозь", ev.get("t_pierce"), "#6a1b9a"),
             ("вернулся к центру", ev.get("ret_center"), "#ef6c00")]
    for name, tj, col in marks:
        if tj is None:
            continue
        tp = pd.Timestamp(minutes[tj])
        ax.axvline(tp, color=col, lw=0.9, alpha=0.55)
        k = tj - a
        if 0 <= k < len(px):
            ax.annotate(name, (tp, px[k]), textcoords="offset points",
                        xytext=(0, 12 if up else -18), fontsize=7, color=col, rotation=0)
    if ev.get("exc_t") is not None:
        te = pd.Timestamp(minutes[ev["exc_t"]])
        ye = close[ev["exc_t"]]
        ax.plot([te], [ye], "o", color="#ef6c00", ms=4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
    ax.set_title(f"{label_extra} {r.side} | центр {r.center:.0f}$ | сила {r.max_loc:.0f} | "
                 f"{fmt_t(r.ts)} UTC | {r['typ'].replace('_',' ')}", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def timeline_rows(r, ev, step=10):
    i_c = ev.get("t_center") if ev.get("t_center") is not None else ev.get("t_edge")
    if i_c is None:
        return []
    a, b = max(0, i_c - 30), min(len(close), i_c + 91)
    event_at = {}
    for name, tj in (("касание края", ev.get("t_edge")), ("вход в центр", ev.get("t_center")),
                     ("прошил насквозь", ev.get("t_pierce")), ("вернулся к центру", ev.get("ret_center"))):
        if tj is not None:
            event_at[tj] = name
    out = []
    j = a
    while j < b:
        note = event_at.get(j, "")
        out.append((fmt_t(minutes[j]), f"{close[j]:.1f}", note))
        j += step
    for tj, name in event_at.items():
        if a <= tj < b and tj % step != a % step:
            out.append((fmt_t(minutes[tj]), f"{close[tj]:.1f}", name))
    out.sort(key=lambda x: x[0])
    return out

# ---------------- HTML / MD assembly ----------------
cases_df = pd.DataFrame(rows_out)
type_order = ["зашёл_и_вернулся", "прошил_но_вернулся", "прошил_и_ушёл", "зашёл_и_задержался"]
type_titles = {
    "зашёл_и_вернулся": "A. Зашёл в скопление и вернулся к его центру (доминирующий тип)",
    "прошил_но_вернулся": "B. Прошил скопление насквозь, но за час всё равно вернулся к центру",
    "прошил_и_ушёл": "C. Прошил насквозь и ушёл дальше (скопление не удержало)",
    "зашёл_и_задержался": "D. Зашёл и остался внутри (не вернулся за час)",
}
html = ["""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Поведение цены у скоплений ликвидаций</title>
<style>body{font-family:Segoe UI,Arial;max-width:1100px;margin:12px auto;padding:0 16px;background:#fafafa;color:#222}
h1{font-size:22px}h2{margin-top:34px;border-bottom:2px solid #ccc;padding-bottom:4px}
.case{background:#fff;border:1px solid #ddd;border-radius:8px;margin:18px 0;padding:12px}
.case img{max-width:100%;border:1px solid #eee}
table.tl{font-size:12px;border-collapse:collapse;margin-top:6px}
table.tl td,table.tl th{border:1px solid #e0e0e0;padding:2px 8px}
.ev{color:#d81b60;font-weight:600}meta{color:#666;font-size:13px}
</style></head><body>"""]
html.append("""<h1>Как цена забирала ликвидации и возвращалась: все случаи из данных</h1>
<p><b>Данные:</b> 154 снимка карты ликвидаций (26–29.10.2026), поминутная цена BTC/USDT (Binance спот).
Для каждого снимка найдены все скопления локальных коэффициентов ≥1 (полоса уровней, идущих подряд).
Ниже — <b>каждый случай</b>, когда цена <b>заходила внутрь скопления</b> (доходила до его самой плотной точки),
и каждый случай <b>отбоя у стены</b> (резкий скачок локального коэффициента) без захода внутрь.</p>
<p><b>Как читать график:</b> цветная полоса — зона скопления (красная — скопление НАД ценой, шортов; зелёная — ПОД ценой, лонгов).
Пунктир — центр скопления. Вертикальные линии — моменты: касание края, вход в центр, проход насквозь, возврат к центру.
Время на графиках — UTC; на Binance в правом верхнем углу обычно UTC, так что можно найти эту же свечу на реальном графике
(дата+время указаны в заголовке каждого случая).</p>""")

summ = cases_df.copy()
summ["от"] = summ.apply(lambda r: fmt_t(minutes[r.t_edge]) if r.t_edge == r.t_edge and r.t_edge is not None else "", axis=1)
html.append(f"<p><b>Итог:</b> {len(ent)} записей о заходах внутрь → <b>{len(cases_df)} уникальных событий</b> "
            f"(дубли с соседних снимков сгруппированы, число повторов у события — в скобках). "
            f"Плюс {len(eb)} случаев отбоя у стены без захода.</p>")
html.append("<table class='tl'><tr><th>#</th><th>время (UTC)</th><th>сторона</th><th>центр, $</th><th>сила</th>"
            "<th>расст. от цены, $</th><th>тип</th><th>дубли</th></tr>")
for r in cases_df.itertuples():
    html.append(f"<tr><td>{r.case_id}</td><td>{fmt_t(r.ts)}</td><td>{r.side}</td><td>{r.center:.0f}</td>"
                f"<td>{r.max_loc:.0f}</td><td>{r.dist:+.0f}</td><td>{r.typ.replace('_',' ')}</td><td>{r.dups}</td></tr>")
html.append("</table>")

md = ["# Поведение цены у скоплений ликвидаций — все случаи (26–29.10.2026, UTC)\n",
      f"Записей о заходах внутрь: {len(ent)} → уникальных событий: {len(cases_df)}. "
      f"Отбоев у стены без захода: {len(eb)}.\n",
      "Формат: время UTC, потом цена. «зона» — границы скопления, «центр» — самая плотная точка.\n"]

for typ in type_order:
    sub = cases_df[cases_df.typ == typ].sort_values("ts")
    if not len(sub):
        continue
    html.append(f"<h2 id='{typ}'>{type_titles[typ]} — {len(sub)} случаев</h2>")
    md.append(f"\n## {type_titles[typ]} — {len(sub)} случаев\n")
    for r in sub.itertuples():
        ev = {k: (None if pd.isna(getattr(r, k)) else (int(getattr(r, k)) if k.startswith("t_") or k=="ret_center" or k=="exc_t" else getattr(r, k))) for k in ("t_edge", "t_center", "t_pierce", "ret_center", "exc", "exc_t", "p15", "p30", "p60")}
        png = make_chart(pd.Series(r._asdict()), ev, None,
                         label_extra=f"#{r.case_id} (повтор на снимках: {r.dups})")
        html.append(f"<div class='case' id='case{r.case_id}'>")
        html.append(f"<meta><b>#{r.case_id}</b> · {fmt_t(r.ts)} UTC · скопление <b>{r.side}</b> "
                    f"(над ценой = шорты, под ценой = лонги) · зона <b>{r.lo:.0f}–{r.hi:.0f} $</b> · "
                    f"центр <b>{r.center:.0f} $</b> · сила {r.max_loc:.0f} · расстояние от цены на снимке {r.dist:+.0f} $ · "
                    f"повторов на соседних снимках: {r.dups}</meta>")
        html.append(f"<img src='data:image/png;base64,{png}'>")
        tl = timeline_rows(pd.Series(r._asdict()), ev)
        html.append("<table class='tl'><tr><th>время UTC</th><th>цена</th><th>что произошло</th></tr>")
        for t_, p_, n_ in tl:
            cls = " class='ev'" if n_ else ""
            html.append(f"<tr><td>{t_}</td><td>{p_}</td><td{cls}>{n_}</td></tr>")
        html.append("</table></div>")
        md.append(f"\n### #{r.case_id} — {fmt_t(r.ts)} UTC — {r.side} зона {r.lo:.0f}–{r.hi:.0f}$, "
                  f"центр {r.center:.0f}$, сила {r.max_loc:.0f}, {r.typ.replace('_',' ')} (повторов: {r.dups})")
        for t_, p_, n_ in tl:
            md.append(f"- {t_}  {p_}$" + (f"  ← **{n_}**" if n_ else ""))

# edge-bounce section
if len(eb):
    html.append(f"<h2 id='wall'>E. Отбой у СТЕНЫ (резкий скачок локального) — цена коснулась края, но внутрь НЕ зашла — {len(eb)} случаев</h2>")
    md.append(f"\n## E. Отбой у стены без захода внутрь — {len(eb)} случаев\n")
    for k, r in enumerate(eb.itertuples(), 1):
        ev = case_events(r)
        ev["typ"] = "отбой_у_стены"
        r_s = pd.Series({**r._asdict(), "typ": "отбой_у_стены"})
        png = make_chart(r_s, ev, None, label_extra=f"E{k}")
        deep = ev.get("deep", float("nan"))
        html.append(f"<div class='case'><meta><b>E{k}</b> · {fmt_t(r.ts)} UTC · стена {r.side} · "
                    f"зона {r.lo:.0f}–{r.hi:.0f}$ · центр {r.center:.0f}$ · сила {r.max_loc:.0f} · "
                    f"скачок локального ×{getattr(r, 'wall_ratio', float('nan')):.0f} · "
                    f"цена зашла в зону на {deep:.0f}$ и не дошла до центра</meta>")
        html.append(f"<img src='data:image/png;base64,{png}'></div>")
        md.append(f"- E{k} {fmt_t(r.ts)} UTC стена {r.side} зона {r.lo:.0f}–{r.hi:.0f}$ центр {r.center:.0f}$ "
                  f"сила {r.max_loc:.0f}: цена вошла в зону на {deep:.0f}$, центр ({r.center:.0f}$) не взят, "
                  f"отбой у края.")

# cascade special
html.append("<h2 id='cascade'>F. Особый случай: цепочка потребления 29.10 (вечерний обвал) — цена съела 6+ скоплений подряд</h2>")
md.append("\n## F. Цепочка 29.10 вечер: цена съела скопления одно за другим вниз\n")
snap_c = cases_df[(cases_df.ts >= "2026-10-29") & (cases_df.side == "DOWN")].sort_values("ts")
i_a, i_b = idx_of(pd.Timestamp("2026-10-29 15:30", tz="UTC")), idx_of(pd.Timestamp("2026-10-29 23:59", tz="UTC"))
fig, ax = plt.subplots(figsize=(11, 5), dpi=90)
t = minutes[i_a:i_b]
ax.plot(t, close[i_a:i_b], lw=1.3, color="#1f77b4", label="цена BTC")
for r in cases_df[(cases_df.ts >= "2026-10-29") & (cases_df.side == "DOWN")].itertuples():
    ax.axhspan(r.lo, r.hi, color="#2e7d32", alpha=0.08)
    ax.axhline(r.center, color="#2e7d32", lw=0.6, ls=":", alpha=0.7)
    if r.t_center == r.t_center and r.t_center is not None and i_a <= r.t_center < i_b:
        tp = pd.Timestamp(minutes[int(r.t_center)])
        ax.plot([tp], [r.center], "v", color="#c62828", ms=7)
        ax.annotate(f"{r.center:.0f}", (tp, r.center), fontsize=7, xytext=(4, -10), textcoords="offset points")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_title("29.10.2026 UTC: движение вниз через цепочку скоплений лонгов (зелёные зоны, ▼ = момент взятия центра)")
ax.grid(alpha=0.25); fig.tight_layout()
buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
png_c = base64.b64encode(buf.getvalue()).decode()
html.append(f"<img src='data:image/png;base64,{png_c}'>")
md.append("График в HTML-версии (artifacts/behavior_cases.html). Точки ▼ — моменты, когда цена брала центр каждого скопления.")

html.append("</body></html>")

with open(os.path.join(OUT, "behavior_cases.html"), "w") as f:
    f.write("\n".join(html))
with open(os.path.join(OUT, "behavior_cases.md"), "w") as f:
    f.write("\n".join(md))

summ_cols = cases_df[["case_id", "ts", "side", "lo", "hi", "center", "max_loc", "dist", "typ", "dups",
                      "t_edge", "t_center", "t_pierce", "ret_center", "p15", "p30", "p60"]]
summ_cols.to_csv(os.path.join(OUT, "cases_overview.csv"), index=False)
print(f"written {OUT}/behavior_cases.html, .md, cases_overview.csv; cases={len(cases_df)} wall-bounce={len(eb)}")
