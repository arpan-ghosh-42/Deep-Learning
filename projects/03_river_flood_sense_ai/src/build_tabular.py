"""
Build the tabular flood-forecasting dataset for West Bengal.

One row = one gauge on one day. Every value in a row is knowable on that day: observed
weather and river state up to today, plus TOMORROW's forecast weather (during training we
substitute tomorrow's reanalysis, i.e. what a perfect forecast would have said).

Target
------
flood_next3 = 1 if GloFAS discharge at that gauge breaches its bankfull proxy - the
station's own 98th-percentile daily discharge - on any of the next 3 days. Defining
"flood" per river is essential: 2,000 m3/s is a drought on the Ganga and a disaster on
the Ajay.

Feature engineering lives in features.py so training and live prediction cannot drift.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from features import add_label, engineer


def load(folder, sid):
    p = folder / f"{sid}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def build_station(g):
    w = load(C.RAW_WEATHER, g["id"])
    q = load(C.RAW_DISCHARGE, g["id"])
    if w is None or q is None:
        print(f"  ! skipping {g['id']} (missing weather or discharge)")
        return None

    up = g.get("upstream")
    uw = load(C.RAW_WEATHER, up) if up else None
    uq = load(C.RAW_DISCHARGE, up) if up else None

    merged = w.merge(q[["date", "river_discharge"]], on="date", how="inner")
    thr = pd.to_numeric(merged["river_discharge"], errors="coerce").quantile(C.BANKFULL_Q)

    df = engineer(w, q, uw, uq, bankfull=thr)
    df = add_label(df, thr, C.LEAD_DAYS)

    df.insert(0, "station_id", g["id"])
    df.insert(1, "station_name", g["name"])
    df.insert(2, "river", g["river"])
    df.insert(3, "district", g["district"])
    df["lat"] = g["lat"]
    df["lon"] = g["lon"]
    return df


def main():
    frames = []
    print("Building tabular dataset ...\n")
    for g in C.GAUGES:
        d = build_station(g)
        if d is None:
            continue
        pos = np.nansum(d["flood_next3"].to_numpy())
        print(f"  {g['id']:<22} {g['river']:<13} {len(d):5d} days  "
              f"bankfull={d['bankfull_threshold'].iloc[0]:9.1f} m3/s  "
              f"flood days={int(pos):4d} ({pos/len(d)*100:4.1f}%)")
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["flood_next3", "discharge"])
    # the first 30 days of each station have incomplete antecedent windows
    df = df[df.groupby("station_id").cumcount() >= 30].reset_index(drop=True)

    out = C.PROC / "flood_tabular.csv"
    df.to_csv(out, index=False)
    print(f"\nRows {len(df):,}   stations {df.station_id.nunique()}   "
          f"{df.date.min().date()} -> {df.date.max().date()}")
    print(f"Positive rate {df.flood_next3.mean()*100:.2f}%")
    print(f"Wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")

    # sanity check against real West Bengal flood episodes
    print("\nFlagged flood days during known West Bengal flood episodes:")
    for label, a, b in [("Sep-Oct 2024 (DVC release + depression)", "2024-09-14", "2024-10-05"),
                        ("Aug 2021 (Damodar basin)", "2021-07-28", "2021-08-10"),
                        ("Oct 2021 (post-monsoon depression)", "2021-09-28", "2021-10-08"),
                        ("Aug 2017 (north Bengal + Malda)", "2017-08-10", "2017-08-25")]:
        w = df[(df.date >= a) & (df.date <= b)]
        if len(w) == 0:
            continue
        hit = w.groupby("station_name").flood_next3.max()
        names = ", ".join(sorted(hit[hit > 0].index))
        print(f"  {label:<42} {int(hit.sum()):2d} stations: {names[:110]}")


if __name__ == "__main__":
    main()
