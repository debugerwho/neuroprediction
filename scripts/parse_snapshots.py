#!/usr/bin/env python3
"""Parse magnet calc_*.txt / collisions_*.txt snapshots into a CSV panel.

Outputs (in .cache/parsed/):
  snapshots.csv   — one row per calc file: timestamps, book stats, dividers per window
  levels.csv      — one row per (snapshot, window, side, level): кум_коэфф, лок_коэфф
  collisions.csv  — one row per (collision file, window, side, level): usd, reason
"""
import re
import os
import csv
import glob

SRC = ".cache/snapshots"
DST = ".cache/parsed"
os.makedirs(DST, exist_ok=True)

SEC_RE = re.compile(r"^===== Секция: (.+?) \| Разделитель: ([\d.]+) =====")
BOOK_RE = re.compile(
    r"^Стакан: book_BTCUSDT_mobchart_spot_(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})-(\d{3})Z"
)
BOOKSTAT_RE = re.compile(
    r"^Стакан: (\d+) бидов \(([\d.]+)-([\d.]+)\), (\d+) асков \(([\d.]+)-([\d.]+)\)"
)
CALC_RE = re.compile(r"^Расчёт: (\d{4}-\d{2}-\d{2} \d{2}:\d{2})")
COLL_RE = re.compile(r"^Коллизии: (\d{4}-\d{2}-\d{2} \d{2}:\d{2})")
LIQ_RE = re.compile(r"^Ликвидации: (liquidations_\S+)")
NUM_RE = re.compile(r"^([\d.]+);([\d.eE+-]+);([\d.eE+-]+)")

WINDOWS = ["1 день", "7 день", "30 день", "90 день", "180 день", "1 Год"]
WKEY = {
    "1 день": "w1d",
    "7 день": "w7d",
    "30 день": "w30d",
    "90 день": "w90d",
    "180 день": "w180d",
    "1 Год": "w1y",
}


def parse_calc(path):
    meta = {}
    dividers = {}
    rows = []
    window = side = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip("\r\n")
            m = CALC_RE.match(line)
            if m:
                meta["ts_local"] = m.group(1)
                continue
            m = LIQ_RE.match(line)
            if m:
                meta["liq_file"] = m.group(1)
                continue
            m = BOOK_RE.match(line)
            if m:
                meta["ts_utc"] = (
                    f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}.{m.group(5)}"
                )
                continue
            m = BOOKSTAT_RE.match(line)
            if m:
                meta["bid_count"] = int(m.group(1))
                meta["bid_lo"] = float(m.group(2))
                meta["bid_hi"] = float(m.group(3))  # best bid
                meta["ask_count"] = int(m.group(4))
                meta["ask_lo"] = float(m.group(5))  # best ask
                meta["ask_hi"] = float(m.group(6))
                continue
            m = SEC_RE.match(line)
            if m:
                window = m.group(1)
                dividers[window] = float(m.group(2))
                side = None
                continue
            if line.startswith("--- ВВЕРХ"):
                side = "UP"
                continue
            if line.startswith("--- ВНИЗ"):
                side = "DOWN"
                continue
            if line.startswith("уровень"):
                continue
            m = NUM_RE.match(line)
            if m and window and side:
                rows.append(
                    (window, side, float(m.group(1)), float(m.group(2)), float(m.group(3)))
                )
    return meta, dividers, rows


def parse_coll(path):
    meta = {}
    dividers = {}
    rows = []
    window = side = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip("\r\n")
            m = COLL_RE.match(line)
            if m:
                meta["ts_local"] = m.group(1)
                continue
            m = LIQ_RE.match(line)
            if m:
                meta["liq_file"] = m.group(1)
                continue
            m = BOOK_RE.match(line)
            if m:
                meta["ts_utc"] = (
                    f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}.{m.group(5)}"
                )
                continue
            m = SEC_RE.match(line)
            if m:
                window = m.group(1)
                dividers[window] = float(m.group(2))
                side = None
                continue
            if line.startswith("--- ВВЕРХ"):
                side = "UP"
                continue
            if line.startswith("--- ВНИЗ"):
                side = "DOWN"
                continue
            if not line or line.startswith("уровень") or line.startswith("(нет"):
                continue
            parts = line.split(";")
            if len(parts) >= 3 and window and side:
                try:
                    rows.append(
                        (window, side, float(parts[0]), float(parts[1]), parts[2].strip())
                    )
                except ValueError:
                    pass
    return meta, dividers, rows


def main():
    snap_rows = []
    level_rows = []
    for path in sorted(glob.glob(os.path.join(SRC, "calc_*.txt"))):
        meta, dividers, rows = parse_calc(path)
        fname = os.path.basename(path)
        bb = meta.get("bid_hi")
        ba = meta.get("ask_lo")
        rec = {
            "file": fname,
            "ts_local": meta.get("ts_local"),
            "ts_utc": meta.get("ts_utc"),
            "liq_file": meta.get("liq_file"),
            "bid_count": meta.get("bid_count"),
            "bid_lo": meta.get("bid_lo"),
            "best_bid": bb,
            "ask_count": meta.get("ask_count"),
            "best_ask": ba,
            "ask_hi": meta.get("ask_hi"),
            "mid": (bb + ba) / 2 if (bb is not None and ba is not None) else None,
            "spread": (ba - bb) if (bb is not None and ba is not None) else None,
        }
        for w in WINDOWS:
            rec[WKEY[w]] = dividers.get(w)
        snap_rows.append(rec)
        ts = rec["ts_utc"]
        for window, side, level, cum, loc in rows:
            level_rows.append((fname, ts, window, side, level, cum, loc))

    with open(os.path.join(DST, "snapshots.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(snap_rows[0].keys()))
        w.writeheader()
        w.writerows(snap_rows)

    with open(os.path.join(DST, "levels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "ts_utc", "window", "side", "level", "cum", "loc"])
        w.writerows(level_rows)

    coll_rows = []
    for path in sorted(glob.glob(os.path.join(SRC, "collisions_*.txt"))):
        meta, dividers, rows = parse_coll(path)
        fname = os.path.basename(path)
        ts = meta.get("ts_utc")
        for window, side, level, usd, reason in rows:
            coll_rows.append((fname, ts, window, side, level, usd, reason))

    with open(os.path.join(DST, "collisions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "ts_utc", "window", "side", "level", "usd", "reason"])
        w.writerows(coll_rows)

    print(f"snapshots={len(snap_rows)} levels={len(level_rows)} collisions={len(coll_rows)}")


if __name__ == "__main__":
    main()
