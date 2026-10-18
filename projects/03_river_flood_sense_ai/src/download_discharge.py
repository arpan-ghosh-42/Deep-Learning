"""
Download daily river discharge for every West Bengal gauge and dam outlet.

Source: Open-Meteo Flood API, serving the Copernicus Emergency Management Service
GloFAS v4 river-discharge reanalysis on a 0.05 degree river network (free, no API key).

Grid snapping
-------------
GloFAS resolves rivers on a coarse grid, so a nominal town coordinate often lands on a
minor cell rather than the main channel: Farakka reads 2 m3/s on its raw coordinate and
~14,000 m3/s one cell away. Every point is therefore SNAPPED - we probe a 5x5 block of
cells (+/- 0.1 deg, ~11 km) over one monsoon season and keep the cell carrying the most
water, then download the full record for that cell.

Probing is batched: the API accepts many coordinates per call, so all 28 stations are
snapped in a handful of requests instead of 28.

Known limitation
----------------
GloFAS represents the Bhagirathi-Hooghly distributary system and the tidal Damodar
poorly - flow there is split by barrage operation, which a global model does not know.
Those stations keep genuine rainfall-driven dynamics but small absolute discharge, which
is why flood labels are defined per station as exceedance of that station's OWN high
quantile rather than as an absolute cumec value. Every snap is logged with its distance
and mean flow in _snap_metadata.csv so the choice stays auditable.
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from net import get_json

SNAP_OFFSETS = [-0.10, -0.05, 0.0, 0.05, 0.10]      # 5x5 block, ~ +/- 11 km
SNAP_PROBE = ("2019-06-01", "2019-10-31")           # one full monsoon season
BATCH = 100                                          # coordinates per probe request


def snap_all(points):
    """Batch-probe a 5x5 block around every point; return {id: (lat, lon, mean_q)}."""
    cand, owner = [], []
    for pt in points:
        for dy in SNAP_OFFSETS:
            for dx in SNAP_OFFSETS:
                cand.append((round(pt["lat"] + dy, 4), round(pt["lon"] + dx, 4)))
                owner.append(pt["id"])

    best = {}
    for s in range(0, len(cand), BATCH):
        chunk, own = cand[s:s + BATCH], owner[s:s + BATCH]
        d = get_json(C.FLOOD_URL, {
            "latitude": ",".join(str(a) for a, _ in chunk),
            "longitude": ",".join(str(b) for _, b in chunk),
            "daily": "river_discharge",
            "start_date": SNAP_PROBE[0], "end_date": SNAP_PROBE[1],
        })
        rows = d if isinstance(d, list) else [d]
        for sid, r in zip(own, rows):
            vals = [v for v in r["daily"]["river_discharge"] if v is not None]
            if not vals:
                continue
            m = sum(vals) / len(vals)
            if sid not in best or m > best[sid][2]:
                best[sid] = (r["latitude"], r["longitude"], m)
        print(f"  probed {min(s + BATCH, len(cand))}/{len(cand)} candidate cells")
        time.sleep(4.0)
    return best


def fetch_series(lat, lon):
    d = get_json(C.FLOOD_URL, {
        "latitude": lat, "longitude": lon,
        "daily": "river_discharge",
        "start_date": C.START_DATE, "end_date": C.END_DATE,
    })
    return pd.DataFrame({
        "date": pd.to_datetime(d["daily"]["time"]),
        "river_discharge": d["daily"]["river_discharge"],
    })


def main():
    points = C.GAUGES + C.DAMS
    dam_ids = {d["id"] for d in C.DAMS}
    print(f"Snapping {len(points)} points to the GloFAS river network "
          f"(5x5 block, {SNAP_PROBE[0]}..{SNAP_PROBE[1]}) ...")
    snap = snap_all(points)

    print(f"\nDownloading {C.START_DATE} -> {C.END_DATE}\n")
    rows = []
    for i, pt in enumerate(points, 1):
        slat, slon, mean_q = snap[pt["id"]]
        shift_km = ((slat - pt["lat"]) ** 2 + (slon - pt["lon"]) ** 2) ** 0.5 * 111
        out = C.RAW_DISCHARGE / f"{pt['id']}.csv"
        if out.exists() and out.stat().st_size > 0:
            df = pd.read_csv(out)
            status = "cached"
        else:
            df = fetch_series(slat, slon)
            df.to_csv(out, index=False)
            status = "fetched"
            time.sleep(4.0)
        kind = "dam" if pt["id"] in dam_ids else "gauge"
        print(f"[{i:2d}/{len(points)}] {pt['id']:<22} {kind:<5} {status}  "
              f"snap {slat:.3f},{slon:.3f} (+{shift_km:4.1f} km)  "
              f"monsoon mean={mean_q:9.1f} m3/s  {len(df)} days")
        rows.append(dict(id=pt["id"], name=pt["name"], river=pt["river"], kind=kind,
                         req_lat=pt["lat"], req_lon=pt["lon"],
                         snap_lat=slat, snap_lon=slon, snap_shift_km=round(shift_km, 1),
                         monsoon_mean_m3s=round(mean_q, 1),
                         record_mean_m3s=round(pd.to_numeric(df["river_discharge"]).mean(), 1),
                         n_days=len(df)))

    pd.DataFrame(rows).to_csv(C.RAW_DISCHARGE / "_snap_metadata.csv", index=False)
    print(f"\nWrote {len(rows)} series + _snap_metadata.csv to {C.RAW_DISCHARGE}")


if __name__ == "__main__":
    main()
