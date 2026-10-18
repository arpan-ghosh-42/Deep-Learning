"""
Pair every satellite chip with the weather and river discharge that produced it.

This is what makes the project genuinely multimodal rather than two unrelated models
bolted together. Each Sen1Floods11 chip carries a real WGS84 centroid (read from its
GeoTIFF tags) and a real acquisition date (the Sentinel-2 pass date of its flood event),
so for every chip we can go and fetch:

  * 40 days of ERA5 daily weather ending the day after the image was taken
  * 40 days of GloFAS river discharge at the same point

and then run the exact same feature engineering used for the West Bengal training set.
The result is a table where each row has BOTH the image-derived flood extent and the
hydro-meteorological state on that day - which is what the fusion model learns from.

Requests are batched by acquisition date (all chips in one flood event share a date), so
446 chips cost a few dozen API calls rather than 892.

Usage:
    python src/build_multimodal.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from features import engineer
from net import download_file, get_json

WINDOW_DAYS = 40      # history needed for the 30-day antecedent features
BATCH = 12            # coordinates per request (Open-Meteo weights by locations x days x vars)
PAUSE = 8.0

META_URL = f"https://storage.googleapis.com/{C.GCS_BUCKET}/v1.1/Sen1Floods11_Metadata.geojson"


def event_dates():
    """{event_name: acquisition date} from the official Sen1Floods11 metadata."""
    import json
    p = C.RAW_SAT / "Sen1Floods11_Metadata.geojson"
    download_file(META_URL, p)
    gj = json.load(open(p))
    out = {}
    for f in gj["features"]:
        pr = f["properties"]
        out[pr["location"]] = pd.Timestamp(str(pr["s2_date"]).replace("/", "-"))
    # the chips are named "Mekong_*" while the metadata calls the event "Cambodia"
    if "Cambodia" in out and "Mekong" not in out:
        out["Mekong"] = out["Cambodia"]
    return out


def batched(url, coords, extra):
    """One request for up to BATCH coordinates; returns a list of per-location dicts."""
    d = get_json(url, {
        "latitude": ",".join(f"{a:.5f}" for a, _ in coords),
        "longitude": ",".join(f"{b:.5f}" for _, b in coords),
        **extra,
    })
    return d if isinstance(d, list) else [d]


def main():
    meta = pd.read_csv(C.PROC / "chip_meta.csv")
    dates = event_dates()
    missing = sorted(set(meta.event) - set(dates))
    if missing:
        raise SystemExit(f"no acquisition date for events: {missing}")
    meta["image_date"] = meta.event.map(dates)
    pred_cols = [c for c in meta.columns if c.startswith("pred_water_frac")]
    if not pred_cols:
        raise SystemExit("chip_meta.csv has no pred_water_frac_* column - "
                         "run src/train_cnn.py first")
    print(f"image branches available: {', '.join(pred_cols)}")

    print(f"Pairing {len(meta)} chips across {meta.event.nunique()} events with "
          f"{WINDOW_DAYS}d of weather + discharge\n")

    rows = []
    for event, grp in meta.groupby("event"):
        d1 = grp.image_date.iloc[0]
        start = (d1 - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
        end = (d1 + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        grp = grp.reset_index(drop=True)
        print(f"  {event:<12} {len(grp):3d} chips  image {d1.date()}  window {start}..{end}")

        for s in range(0, len(grp), BATCH):
            sub = grp.iloc[s:s + BATCH]
            coords = list(zip(sub.lat, sub.lon))

            wx = batched(C.ARCHIVE_URL, coords, {
                "start_date": start, "end_date": end,
                "daily": ",".join(C.DAILY_WEATHER_VARS), "timezone": "UTC"})
            time.sleep(PAUSE)
            qx = batched(C.FLOOD_URL, coords, {
                "start_date": start, "end_date": end, "daily": "river_discharge"})
            time.sleep(PAUSE)

            for (_, chip), w, q in zip(sub.iterrows(), wx, qx):
                wdf = pd.DataFrame({"date": pd.to_datetime(w["daily"]["time"])})
                for v in C.DAILY_WEATHER_VARS:
                    wdf[v] = w["daily"][v]
                qdf = pd.DataFrame({"date": pd.to_datetime(q["daily"]["time"]),
                                    "river_discharge": q["daily"]["river_discharge"]})
                feat = engineer(wdf, qdf)
                # the row for the image date itself
                row = feat[feat.date == d1]
                if row.empty:
                    row = feat.iloc[[-2]]
                row = row.iloc[0].to_dict()
                row.pop("date", None)
                rows.append(dict(chip=chip.chip, event=event,
                                 image_date=d1.strftime("%Y-%m-%d"),
                                 lat=chip.lat, lon=chip.lon,
                                 water_frac=chip.water_frac,
                                 perm_water_frac=chip.perm_water_frac,
                                 flood_frac=chip.flood_frac,
                                 is_flooded=chip.is_flooded,
                                 split=chip.get("split", "train"),
                                 # carry every U-Net prediction column through so the
                                 # fusion head can pick its image branch
                                 **{c: chip[c] for c in pred_cols},
                                 **row))
            print(f"    fetched {min(s + BATCH, len(grp))}/{len(grp)}")

    out = pd.DataFrame(rows)
    p = C.PROC / "multimodal.csv"
    out.to_csv(p, index=False)
    print(f"\nWrote {p}  ({len(out)} rows x {len(out.columns)} cols)")
    print(f"Flooded chips: {out.is_flooded.sum()}/{len(out)}")
    print("\nMean conditions, flooded vs not:")
    cols = ["precipitation_sum", "precip_7d", "precip_30d", "api_15d",
            "soil_moisture_0_to_100cm_mean", "discharge", "water_frac", "flood_frac"]
    print(out.groupby("is_flooded")[[c for c in cols if c in out]].mean().T.to_string())


if __name__ == "__main__":
    main()
