"""
River Flood Sense AI - interactive demo.

Run with:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C          # noqa: E402
import predict as P         # noqa: E402

st.set_page_config(page_title="River Flood Sense AI", page_icon="🌊", layout="wide")


@st.cache_resource
def get_unet(mode):
    return P.load_unet(mode)


# cache_resource, not cache_data: this returns DataFrames that we want kept as objects
# rather than round-tripped through JSON (which mangles the date column).
@st.cache_resource(ttl=1800, show_spinner=False)
def get_live(station_id):
    return P.fetch_live(station_id)


@st.cache_data
def chip_catalogue():
    p = C.PROC / "chip_meta.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


st.title("🌊 River Flood Sense AI")
st.caption("Flood risk for West Bengal rivers from a satellite image + today's and "
           "tomorrow's weather. Demo / educational project — not an operational warning "
           "system.")

# ----------------------------------------------------------------- sidebar: station
gauges = {g["id"]: f"{g['name']} — {g['river']} ({g['district']})" for g in C.GAUGES}
with st.sidebar:
    st.header("1 · Location")
    station = st.selectbox("River gauge", list(gauges), format_func=lambda k: gauges[k],
                           index=list(gauges).index("ghatal_silabati"))
    st.caption("Ghatal on the Shilabati and Udaynarayanpur on the Damodar are West "
               "Bengal's most reliably flooded reaches.")
    use_live = st.toggle("Fetch live conditions", value=True,
                         help="Pulls the real current weather and GloFAS river discharge "
                              "for this gauge from Open-Meteo.")

live = None
if use_live:
    try:
        with st.spinner("Fetching live weather and river discharge…"):
            live = get_live(station)
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Live fetch failed: {str(e)[:160]}")

defaults = dict(temp_max_today=33.0, temp_min_today=26.0, humidity_today=80.0,
                rain_today=20.0, temp_max_tomorrow=32.0, temp_min_tomorrow=26.0,
                humidity_tomorrow=82.0, rain_tomorrow=25.0, dam_outflow=None,
                discharge=None)
base_row = None
if live is not None:
    try:
        base_row, _feat, _thr = P.build_row(live)
        defaults.update(
            temp_max_today=float(base_row["temperature_2m_max"]),
            temp_min_today=float(base_row["temperature_2m_min"]),
            humidity_today=float(base_row["relative_humidity_2m_mean"]),
            rain_today=float(base_row["precipitation_sum"]),
            temp_max_tomorrow=float(base_row["temperature_2m_max_t1"]),
            temp_min_tomorrow=float(base_row["temperature_2m_min_t1"]),
            humidity_tomorrow=float(base_row["relative_humidity_2m_mean_t1"]),
            rain_tomorrow=float(base_row["precipitation_sum_t1"]),
            dam_outflow=(float(base_row["dam_outflow"])
                         if base_row.get("has_upstream_dam") and
                         pd.notna(base_row.get("dam_outflow")) else None),
            discharge=float(base_row["discharge"]) if pd.notna(base_row["discharge"]) else None)
    except Exception as e:  # noqa: BLE001
        st.sidebar.warning(f"Could not build feature row: {str(e)[:160]}")

with st.sidebar:
    st.header("2 · Conditions")
    st.caption("Pre-filled from live data — change any value to explore a scenario.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Today**")
        tmax0 = st.number_input("Max temp °C", value=defaults["temp_max_today"], step=0.5)
        tmin0 = st.number_input("Min temp °C", value=defaults["temp_min_today"], step=0.5)
        rh0 = st.number_input("Humidity %", value=defaults["humidity_today"], step=1.0)
        rain0 = st.number_input("Rainfall mm", value=defaults["rain_today"], step=5.0,
                                min_value=0.0)
    with c2:
        st.markdown("**Tomorrow**")
        tmax1 = st.number_input("Max temp °C ", value=defaults["temp_max_tomorrow"], step=0.5)
        tmin1 = st.number_input("Min temp °C ", value=defaults["temp_min_tomorrow"], step=0.5)
        rh1 = st.number_input("Humidity % ", value=defaults["humidity_tomorrow"], step=1.0)
        rain1 = st.number_input("Rainfall mm ", value=defaults["rain_tomorrow"], step=5.0,
                                min_value=0.0)

    st.markdown("**River & dam**")
    dam_default = defaults["dam_outflow"]
    dam = st.number_input(
        "Upstream dam outflow (m³/s)",
        value=float(dam_default) if dam_default is not None else 0.0,
        step=50.0, min_value=0.0,
        help="Proxy for a DVC / barrage release. Raise it to simulate a big release.",
        disabled=dam_default is None)
    if dam_default is None:
        st.caption("This gauge sits on an unregulated river — no controlling dam.")

st.sidebar.header("3 · Satellite image")
catalogue = chip_catalogue()
src = st.sidebar.radio("Image source", ["Sample from dataset", "Upload my own", "None"],
                       index=0, label_visibility="collapsed")

image_path = None
if src == "Upload my own":
    up = st.sidebar.file_uploader("Satellite image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if up:
        tmp = C.REPORTS / f"_upload{Path(up.name).suffix}"
        tmp.write_bytes(up.getbuffer())
        image_path = tmp
    known_perm = None
elif src == "Sample from dataset" and not catalogue.empty:
    ev = st.sidebar.selectbox("Flood event", sorted(catalogue.event.unique()),
                              index=sorted(catalogue.event.unique()).index("India")
                              if "India" in set(catalogue.event) else 0)
    sub = catalogue[catalogue.event == ev].sort_values("flood_frac", ascending=False)
    lbl = {r.chip: (f"{'FLOODED' if r.is_flooded else 'normal '} · "
                    f"water {r.water_frac*100:4.1f}% · excess {r.flood_frac*100:4.1f}%")
           for r in sub.itertuples()}
    chip = st.sidebar.selectbox("Chip", list(lbl), format_func=lambda k: f"{k} — {lbl[k]}")
    image_path = ROOT / "datasets/processed/images/preview" / f"{chip}.png"
    known_perm = float(sub.set_index("chip").loc[chip, "perm_water_frac"])
else:
    known_perm = None

perm = st.sidebar.slider(
    "Normal (dry-season) river coverage of the scene", 0.0, 0.40,
    float(known_perm if known_perm is not None else P.DEFAULT_PERM_WATER), 0.005,
    help="How much of this view is water even in a normal year. Flood extent is measured "
         "as water ABOVE this baseline, so a naturally wide river is not called a flood.")

# ----------------------------------------------------------------------- run models
overrides = dict(temp_max_today=tmax0, temp_min_today=tmin0, humidity_today=rh0,
                 rain_today=rain0, temp_max_tomorrow=tmax1, temp_min_tomorrow=tmin1,
                 humidity_tomorrow=rh1, rain_tomorrow=rain1,
                 dam_outflow=dam if dam_default is not None else None)

row = None
if live is not None:
    try:
        row, feat, thr = P.build_row(live, overrides=overrides)
    except Exception as e:  # noqa: BLE001
        st.error(f"Feature build failed: {str(e)[:200]}")

water_frac = None
left, right = st.columns([1, 1])

with left:
    st.subheader("Satellite view")
    if image_path and Path(image_path).exists():
        try:
            x, mode, rgb = P.prepare_image(image_path)
            model = get_unet(mode)
            prob, water_frac = P.segment_water(model, x)
            a, b = st.columns(2)
            a.image(rgb, caption="Input image", use_container_width=True)
            b.image(P.overlay(rgb, prob), caption="Detected water (U-Net)",
                    use_container_width=True)
            excess = max(0.0, water_frac - perm)
            m1, m2, m3 = st.columns(3)
            m1.metric("Water in scene", f"{water_frac*100:.1f}%")
            m2.metric("Normal extent", f"{perm*100:.1f}%")
            m3.metric("Excess water", f"{excess*100:.1f}%",
                      delta="flood signal" if excess > C.FLOOD_FRAC_THRESHOLD else "within normal",
                      delta_color="inverse" if excess > C.FLOOD_FRAC_THRESHOLD else "off")
            st.caption(f"Branch: **{mode.upper()}** "
                       f"({'full Sentinel-2, 7 channels' if mode == 'ms' else 'RGB, 3 channels'})")
        except Exception as e:  # noqa: BLE001
            st.error(f"Image analysis failed: {str(e)[:200]}")
    else:
        st.info("Pick a sample chip or upload a satellite image in the sidebar.")

with right:
    st.subheader("Flood assessment")
    if row is None:
        st.warning("No live hydro-meteorological data — enable **Fetch live conditions**.")
    else:
        p_tab, _ = P.tabular_probability(row)
        p_fus = (P.fusion_probability(row, water_frac, perm)
                 if water_frac is not None else None)
        band, note = P.risk_band(p_tab)
        colour = {"SEVERE": "🔴", "HIGH": "🟠", "WATCH": "🟡",
                  "LOW": "🟢", "unknown": "⚪"}[band]

        if p_tab is not None:
            st.metric(f"{colour} 3-day bankfull-breach probability", f"{p_tab*100:.1f}%",
                      delta=band, delta_color="off")
            st.progress(min(max(p_tab, 0.0), 1.0))
            st.caption(note)
        if p_fus is not None:
            st.metric("Fused flood probability (image + weather)", f"{p_fus*100:.1f}%")

        st.divider()
        st.markdown("**Why — the drivers behind this number**")
        drivers = [
            ("River level now", f"{row['discharge']:,.0f} m³/s "
                                f"({row['discharge_norm']*100:.0f}% of bankfull)"),
            ("3-day river rise", f"{row['discharge_chg_3d']:+,.0f} m³/s"),
            ("Rain today → tomorrow", f"{row['precipitation_sum']:.0f} → "
                                      f"{row['precipitation_sum_t1']:.0f} mm"),
            ("Rain last 7 / 30 days", f"{row['precip_7d']:.0f} / {row['precip_30d']:.0f} mm"),
            ("Antecedent index (API)", f"{row['api_15d']:.0f} mm"),
            ("Deep soil moisture", f"{row['soil_moisture_0_to_100cm_mean']:.3f} m³/m³"),
            ("Upstream catchment rain 3d", f"{row['upstream_precip_3d']:.0f} mm"
                if pd.notna(row.get('upstream_precip_3d')) else "—"),
            ("Temp today → tomorrow", f"{row['temperature_2m_max']:.1f} → "
                                      f"{row['temperature_2m_max_t1']:.1f} °C"),
            ("Humidity today → tomorrow", f"{row['relative_humidity_2m_mean']:.0f} → "
                                          f"{row['relative_humidity_2m_mean_t1']:.0f} %"),
        ]
        if row.get("has_upstream_dam"):
            drivers.insert(2, ("Upstream dam outflow",
                               f"{row['dam_outflow']:,.0f} m³/s "
                               f"({row['dam_outflow_chg_1d']:+,.0f} in 24h)"))
        st.dataframe(pd.DataFrame(drivers, columns=["Factor", "Value"]),
                     hide_index=True, use_container_width=True)
        st.caption(f"Bankfull threshold for this gauge: {thr:,.0f} m³/s "
                   f"(98th percentile of its 2005–2026 GloFAS record).")

st.divider()
with st.expander("How this works, and what it is not"):
    st.markdown("""
**Two branches, one score.**

* **Image branch** — a U-Net segments open water in the satellite scene. Flood extent is
  the water it finds *minus* the river's normal dry-season extent, so a naturally wide
  river is not reported as a flood.
* **Hydro-meteorological branch** — gradient boosting over ~50 features: today's and
  tomorrow's temperature, humidity and rainfall; 3/7/15/30-day antecedent rainfall and an
  antecedent precipitation index; soil moisture; current river discharge and its rise
  rate; and the upstream dam's outflow. It predicts whether the gauge breaches its
  bankfull threshold within 3 days.
* **Fusion head** — combines the visible inundation with the hydro-meteorology.

**Honest limits.** Dam-release figures are a GloFAS proxy, not published DVC cusec
readings. GloFAS resolves the Bhagirathi–Hooghly distributaries poorly. Flood labels are
threshold exceedances, not surveyed inundation. The satellite model is trained on
Sen1Floods11 — global flood imagery including a Ganga-basin event — not on a
West-Bengal-only image set. This is a portfolio demo; for real warnings use the
[West Bengal Irrigation & Waterways Department](https://wbiwd.gov.in/) and IMD bulletins.
""")
