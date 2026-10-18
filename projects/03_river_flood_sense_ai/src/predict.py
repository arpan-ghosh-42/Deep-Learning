"""
Inference: turn a satellite image plus today's / tomorrow's conditions into a flood
assessment for a West Bengal river gauge.

Three questions get answered separately, because they are genuinely different questions:

  1. IMAGE      How much of this scene is water right now, and how much of that is
                more water than this reach normally carries?     (U-Net)
  2. FORECAST   Given the weather, the soaked-or-dry catchment, the current river level
                and the upstream dam's outflow, what is the chance this gauge breaches
                bankfull within 3 days?                          (gradient boosting)
  3. FUSED      Combining the visible inundation with the hydro-meteorology, how likely
                is it that this scene is genuinely flooded?      (fusion head)

Live mode pulls the real current state for any configured station from Open-Meteo
(40 days of history + a 3-day forecast, plus GloFAS forecast discharge), so the demo
works on today's actual weather. Any field can be overridden by hand.

CLI:
    python src/predict.py --station ghatal_silabati
    python src/predict.py --station farakka_ganga \
        --image datasets/processed/images/preview/India_1017769.png
    python src/predict.py --station kolkata_hooghly --rain-today 180 --rain-tomorrow 120
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from features import engineer
from net import get_json
from train_cnn import UNet, spectral_indices

# Mean permanent-water fraction across the Ganga/Brahmaputra-basin (India event) chips:
# the share of a typical river scene that is water even in a normal year. Used as the
# default "normal river" baseline when scoring an uploaded image whose JRC layer we do not
# have. The median is near zero because JRC marks little of a braided, channel-shifting
# river as permanent, so the mean is the safer default.
DEFAULT_PERM_WATER = 0.027


# ------------------------------------------------------------------ model loading
def load_unet(channels):
    p = C.MODELS / f"unet_water_{channels}.pt"
    if not p.exists():
        raise SystemExit(f"missing {p} - run: python src/train_cnn.py --channels {channels}")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = UNet(cin=ck["in_ch"], base=ck["base"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def load_pickle(name):
    p = C.MODELS / name
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------ image branch
def prepare_image(path):
    """Return (tensor 1xCxHxW, channel mode, uint8 RGB for display)."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        a = tifffile.imread(path).astype(np.float32)
        if a.ndim == 3 and a.shape[0] >= 12:                    # full Sentinel-2 chip
            sel = np.stack([a[C.S2_BAND_IDX[b]]
                            for b in ("red", "green", "blue", "nir", "swir1")])
            sel = np.clip(sel / 10000.0, 0, 1)
            x = spectral_indices(torch.from_numpy(sel).unsqueeze(0))
            disp = np.zeros((3, *sel.shape[1:]), np.float32)
            for i in range(3):
                lo, hi = np.percentile(sel[i], [2, 98])
                disp[i] = np.clip((sel[i] - lo) / max(hi - lo, 1e-6), 0, 1)
            return x, "ms", (disp.transpose(1, 2, 0) * 255).astype(np.uint8)
        rgb = np.asarray(Image.fromarray(a.astype(np.uint8)).convert("RGB"))
    else:
        rgb = np.asarray(Image.open(path).convert("RGB"))
    x = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
    return x, "rgb", rgb


@torch.no_grad()
def segment_water(model, x, thresh=0.5):
    """Pad to a multiple of 8 (the U-Net has 3 pooling levels), predict, crop back."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    prob = torch.sigmoid(model(x))[0][:h, :w].numpy()
    return prob, float((prob > thresh).mean())


def overlay(rgb, mask, colour=(0, 160, 255), alpha=0.45):
    out = rgb.astype(np.float32).copy()
    m = mask > 0.5
    for c in range(3):
        out[..., c][m] = (1 - alpha) * out[..., c][m] + alpha * colour[c]
    return out.astype(np.uint8)


# ------------------------------------------------------------ hydro-met branch
def bankfull_for(station_id):
    p = C.RAW_DISCHARGE / f"{station_id}.csv"
    q = pd.read_csv(p)["river_discharge"]
    return float(pd.to_numeric(q, errors="coerce").quantile(C.BANKFULL_Q))


def fetch_live(station_id, past_days=40, forecast_days=3):
    """Live weather + GloFAS discharge for a station and its upstream dam."""
    g = C.ALL_POINTS[station_id]

    def wx(lat, lon):
        d = get_json(C.FORECAST_URL, {
            "latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
            "daily": ",".join(C.DAILY_WEATHER_VARS),
            "past_days": past_days, "forecast_days": forecast_days})
        f = pd.DataFrame({"date": pd.to_datetime(d["daily"]["time"])})
        for v in C.DAILY_WEATHER_VARS:
            f[v] = d["daily"][v]
        return f

    def qx(sid):
        snap = pd.read_csv(C.RAW_DISCHARGE / "_snap_metadata.csv").set_index("id")
        lat, lon = snap.loc[sid, "snap_lat"], snap.loc[sid, "snap_lon"]
        d = get_json(C.FLOOD_URL, {"latitude": lat, "longitude": lon,
                                   "daily": "river_discharge",
                                   "past_days": past_days, "forecast_days": forecast_days})
        return pd.DataFrame({"date": pd.to_datetime(d["daily"]["time"]),
                             "river_discharge": d["daily"]["river_discharge"]})

    up = g.get("upstream")
    upt = C.ALL_POINTS[up] if up else None
    return dict(weather=wx(g["lat"], g["lon"]), discharge=qx(station_id),
                up_weather=wx(upt["lat"], upt["lon"]) if upt else None,
                up_discharge=qx(up) if up else None, station=g)


OVERRIDE_MAP = {           # cli/app name -> (weather column, day offset)
    "temp_max_today": ("temperature_2m_max", 0),
    "temp_min_today": ("temperature_2m_min", 0),
    "humidity_today": ("relative_humidity_2m_mean", 0),
    "rain_today": ("precipitation_sum", 0),
    "temp_max_tomorrow": ("temperature_2m_max", 1),
    "temp_min_tomorrow": ("temperature_2m_min", 1),
    "humidity_tomorrow": ("relative_humidity_2m_mean", 1),
    "rain_tomorrow": ("precipitation_sum", 1),
}


def build_row(live, target_date=None, overrides=None):
    """Apply overrides, run the shared feature engineering, return the row for the day."""
    w = live["weather"].copy()
    q = live["discharge"].copy()
    # past_days of history + forecast_days ahead: with forecast_days=3 the frame ends at
    # today+2, so today sits at index len(w)-3 and tomorrow's forecast is still present.
    date = pd.Timestamp(target_date).normalize() if target_date else \
        pd.Timestamp(w["date"].iloc[len(w) - 3]).normalize()

    ov = {k: v for k, v in (overrides or {}).items() if v is not None}
    for key, val in ov.items():
        if key in OVERRIDE_MAP:
            col, off = OVERRIDE_MAP[key]
            w.loc[w["date"] == date + pd.Timedelta(days=off), col] = float(val)
    if ov.get("dam_outflow") is not None and live["up_discharge"] is not None:
        live = {**live, "up_discharge": live["up_discharge"].copy()}
        live["up_discharge"].loc[
            live["up_discharge"]["date"] == date, "river_discharge"] = float(ov["dam_outflow"])
    if ov.get("discharge") is not None:
        q.loc[q["date"] == date, "river_discharge"] = float(ov["discharge"])

    thr = bankfull_for(live["station"]["id"])
    dthr = bankfull_for(live["station"]["upstream"]) if live["station"].get("upstream") else None
    feat = engineer(w, q, live["up_weather"], live["up_discharge"],
                    bankfull=thr, dam_bankfull=dthr)
    feat["lat"] = live["station"]["lat"]
    feat["lon"] = live["station"]["lon"]

    row = feat[feat["date"] == date]
    if row.empty:
        row = feat.iloc[[-3]]
    return row.iloc[0], feat, thr


def tabular_probability(row):
    b = load_pickle("tabular_gbm.pkl")
    if b is None:
        return None, None
    x = np.array([[float(row.get(f, np.nan)) for f in b["features"]]], dtype=np.float32)
    return float(b["model"].predict_proba(x)[0, 1]), b["threshold"]


def fusion_probability(row, water_frac, perm_frac):
    b = load_pickle("fusion.pkl")
    if b is None or water_frac is None:
        return None
    vals = dict(row)
    vals[b["image_col"]] = water_frac
    vals["perm_water_frac"] = perm_frac
    vals["excess_water_frac"] = max(0.0, water_frac - perm_frac)
    x = np.array([[float(vals.get(f, np.nan)) for f in b["features"]]], dtype=float)
    x = np.where(np.isnan(x), b["impute_median"], x)
    return float(b["model"].predict_proba(x)[0, 1])


def risk_band(p):
    if p is None:
        return "unknown", "-"
    if p >= 0.70:
        return "SEVERE", "evacuate-planning territory; verify against IWD bulletins"
    if p >= 0.45:
        return "HIGH", "flooding likely in the next 3 days"
    if p >= 0.20:
        return "WATCH", "elevated - monitor rainfall and dam releases"
    return "LOW", "no bankfull breach indicated"


# ------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="ghatal_silabati",
                    choices=[g["id"] for g in C.GAUGES])
    ap.add_argument("--image", default=None, help="satellite image (PNG/JPEG or S2 GeoTIFF)")
    ap.add_argument("--date", default=None)
    ap.add_argument("--perm-water", type=float, default=None,
                    help=f"normal river coverage of the scene (default {DEFAULT_PERM_WATER})")
    for k in OVERRIDE_MAP:
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=None)
    ap.add_argument("--dam-outflow", type=float, default=None)
    args = ap.parse_args()

    st = C.ALL_POINTS[args.station]
    print(f"\n=== River Flood Sense AI :: {st['name']} ({st['river']}, {st['district']}) ===")
    print("fetching live weather + river discharge ...")
    live = fetch_live(args.station)
    ov = {k: getattr(args, k) for k in OVERRIDE_MAP}
    ov["dam_outflow"] = args.dam_outflow
    row, feat, thr = build_row(live, args.date, ov)

    d = pd.Timestamp(row["date"]).date()
    print(f"\nDate {d}   bankfull threshold {thr:,.0f} m3/s")
    print(f"  temp today      {row['temperature_2m_max']:.1f} / {row['temperature_2m_min']:.1f} C"
          f"      tomorrow {row['temperature_2m_max_t1']:.1f} / {row['temperature_2m_min_t1']:.1f} C")
    print(f"  humidity        {row['relative_humidity_2m_mean']:.0f}%"
          f"                tomorrow {row['relative_humidity_2m_mean_t1']:.0f}%")
    print(f"  rain today      {row['precipitation_sum']:.1f} mm"
          f"            tomorrow {row['precipitation_sum_t1']:.1f} mm")
    print(f"  antecedent rain 7d {row['precip_7d']:.0f} mm   30d {row['precip_30d']:.0f} mm"
          f"   API {row['api_15d']:.0f}")
    print(f"  soil moisture   {row['soil_moisture_0_to_100cm_mean']:.3f} m3/m3")
    print(f"  river discharge {row['discharge']:,.0f} m3/s "
          f"({row['discharge_norm'] * 100:.0f}% of bankfull), "
          f"3-day change {row['discharge_chg_3d']:+,.0f}")
    if row.get("has_upstream_dam"):
        print(f"  dam outflow     {row['dam_outflow']:,.0f} m3/s "
              f"({row['dam_outflow_norm'] * 100:.0f}% of its own high mark), "
              f"1-day change {row['dam_outflow_chg_1d']:+,.0f}")

    water_frac = perm = None
    if args.image:
        perm = args.perm_water if args.perm_water is not None else DEFAULT_PERM_WATER
        x, mode, rgb = prepare_image(args.image)
        model = load_unet(mode)
        prob, water_frac = segment_water(model, x)
        out = C.REPORTS / f"segmentation_{Path(args.image).stem}.png"
        Image.fromarray(overlay(rgb, prob)).save(out)
        print(f"\nIMAGE ({mode} branch, {rgb.shape[1]}x{rgb.shape[0]})")
        print(f"  water covers        {water_frac * 100:.1f}% of the scene")
        print(f"  normal river extent {perm * 100:.1f}%")
        print(f"  excess water        {max(0.0, water_frac - perm) * 100:.1f}%"
              f"  <- the flood signal")
        print(f"  overlay written to  {out}")

    p_tab, thr_tab = tabular_probability(row)
    p_fus = fusion_probability(row, water_frac, perm) if water_frac is not None else None

    print("\n--- ASSESSMENT ---")
    if p_tab is not None:
        band, note = risk_band(p_tab)
        print(f"  3-day bankfull-breach probability : {p_tab * 100:5.1f}%   [{band}] {note}")
    if p_fus is not None:
        print(f"  fused flood probability (image+met): {p_fus * 100:5.1f}%")
    print()


if __name__ == "__main__":
    main()
