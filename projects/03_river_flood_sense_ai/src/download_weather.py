"""
Download daily ERA5 weather for every West Bengal gauge, plus catchment rainfall at each
dam / barrage (upstream rain is what fills the reservoirs that later get released).

Source: Open-Meteo Historical Weather API (ERA5 / ERA5-Land reanalysis, free, no key).
20 daily variables per station - temperature, humidity, rainfall, soil moisture, pressure,
radiation, evapotranspiration - listed in config.DAILY_WEATHER_VARS.

Open-Meteo weights its rate limit by (variables x timesteps), so one 20-variable request
over 21 years trips the per-minute quota on its own. Two things keep us inside the free
tier:

  * requests are split into CHUNK_YEARS blocks and stitched back together locally
  * dam / barrage points fetch ONLY precipitation. Their weather is used purely as
    upstream catchment rainfall (see features.py), so pulling the other 19 variables
    there would be 20x the quota for data nothing reads.
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from net import get_json

CHUNK_YEARS = 3
PAUSE = 11.0
DAM_VARS = ["precipitation_sum"]   # all features.py reads from an upstream point


def date_chunks(start, end, years):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    while s <= e:
        stop = min(pd.Timestamp(year=s.year + years, month=1, day=1) - pd.Timedelta(days=1), e)
        out.append((s.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d")))
        s = stop + pd.Timedelta(days=1)
    return out


def fetch(lat, lon, variables):
    # Quota cost scales with variables x days. A 20-variable pull needs splitting; a
    # single-variable one (the dams) is ~7,900 values and fits comfortably in one request,
    # so don't pay eight sleep intervals for it.
    heavy = len(variables) > 3
    years, pause = (CHUNK_YEARS, PAUSE) if heavy else (100, 1.5)
    frames = []
    for a, b in date_chunks(C.START_DATE, C.END_DATE, years):
        d = get_json(C.ARCHIVE_URL, {
            "latitude": lat, "longitude": lon,
            "start_date": a, "end_date": b,
            "daily": ",".join(variables),
            "timezone": "Asia/Kolkata",
        }, tries=10)
        daily = d["daily"]
        f = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
        for v in variables:
            f[v] = daily[v]
        frames.append(f)
        time.sleep(pause)
    return (pd.concat(frames, ignore_index=True)
            .drop_duplicates("date").sort_values("date").reset_index(drop=True))


def main():
    # Dams first: they need one variable instead of twenty, so they cost almost no quota
    # and complete in seconds. Getting them in early means the upstream-catchment-rainfall
    # features are available even if the slower gauge downloads are still running.
    points = C.DAMS + C.GAUGES
    dam_ids = {d["id"] for d in C.DAMS}
    chunks = date_chunks(C.START_DATE, C.END_DATE, CHUNK_YEARS)
    print(f"ERA5 daily weather for {len(points)} points "
          f"({len(C.DAILY_WEATHER_VARS)} vars at gauges, {len(DAM_VARS)} at dams)")
    print(f"{C.START_DATE} -> {C.END_DATE} in {len(chunks)} chunks of {CHUNK_YEARS}y\n")

    for i, pt in enumerate(points, 1):
        out = C.RAW_WEATHER / f"{pt['id']}.csv"
        if out.exists() and out.stat().st_size > 0:
            print(f"[{i:2d}/{len(points)}] {pt['id']:<22} cached")
            continue
        variables = DAM_VARS if pt["id"] in dam_ids else C.DAILY_WEATHER_VARS
        df = fetch(pt["lat"], pt["lon"], variables)
        df.to_csv(out, index=False)
        annual = df["precipitation_sum"].sum() / (len(df) / 365.25)
        print(f"[{i:2d}/{len(points)}] {pt['id']:<22} {len(df)} days  "
              f"nulls={int(df['precipitation_sum'].isna().sum())}  "
              f"annual rain ~{annual:.0f} mm")

    print(f"\nWrote weather to {C.RAW_WEATHER}")


if __name__ == "__main__":
    main()
