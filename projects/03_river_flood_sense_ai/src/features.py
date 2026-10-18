"""
Shared feature engineering.

build_tabular.py (training), build_multimodal.py (pairing satellite chips with the
weather that produced them) and predict.py (live scoring) all call `engineer()`, so a
feature can never mean one thing at training time and another at inference time.
"""
import numpy as np
import pandas as pd

TOMORROW_VARS = [
    "temperature_2m_max", "temperature_2m_min", "relative_humidity_2m_mean",
    "precipitation_sum", "precipitation_hours", "cloud_cover_mean", "pressure_msl_mean",
]

# Columns produced by engineer(), in the order the models expect them.
UPSTREAM_COLS = ["upstream_precip", "upstream_precip_3d", "upstream_precip_7d"]
DAM_COLS = ["dam_outflow", "dam_outflow_norm", "dam_outflow_chg_1d", "dam_outflow_chg_3d"]


def api_index(rain, k=0.90):
    """Antecedent Precipitation Index: exponentially decayed memory of past rainfall.

    API_t = k * API_{t-1} + rain_t. It is the cheapest usable stand-in for catchment
    storage: rain that fell a week ago still matters, rain from last month mostly does not.
    """
    out = np.zeros(len(rain))
    acc = 0.0
    for i, r in enumerate(np.nan_to_num(np.asarray(rain, dtype=float))):
        acc = acc * k + r
        out[i] = acc
    return out


def engineer(weather, discharge=None, up_weather=None, up_discharge=None,
             bankfull=None, dam_bankfull=None):
    """Build the model feature frame.

    weather      daily weather at the gauge, must contain `date` + config.DAILY_WEATHER_VARS
    discharge    daily river discharge at the gauge (`date`, `river_discharge`)
    up_weather   daily weather over the controlling reservoir's catchment (optional)
    up_discharge daily discharge at the reservoir outlet = release proxy (optional)
    bankfull     the gauge's bankfull threshold; if None it is taken from this record

    Returns the frame with every feature column added. Rows are chronological, and every
    feature at row t uses only information available on day t (plus the t+1 forecast).
    """
    df = weather.sort_values("date").reset_index(drop=True).copy()

    if discharge is not None:
        d = discharge[["date", "river_discharge"]].copy()
        df = df.merge(d, on="date", how="left")
        df["discharge"] = pd.to_numeric(df["river_discharge"], errors="coerce")
        df = df.drop(columns=["river_discharge"])
    else:
        df["discharge"] = np.nan

    # ---- tomorrow's forecast ------------------------------------------------------
    for v in TOMORROW_VARS:
        df[f"{v}_t1"] = df[v].shift(-1) if v in df.columns else np.nan
    df["temp_change_1d"] = df["temperature_2m_max_t1"] - df["temperature_2m_max"]
    df["precip_2day_total"] = df["precipitation_sum"] + df["precipitation_sum_t1"]

    # ---- antecedent catchment wetness ---------------------------------------------
    r = df["precipitation_sum"]
    for k in (3, 7, 15, 30):
        df[f"precip_{k}d"] = r.rolling(k, min_periods=1).sum()
    df["api_15d"] = api_index(r, k=0.90)
    df["rain_days_7d"] = (r > 1.0).rolling(7, min_periods=1).sum()
    df["soil_moist_7d_mean"] = df["soil_moisture_0_to_100cm_mean"].rolling(7, min_periods=1).mean()

    # ---- river state ---------------------------------------------------------------
    q = df["discharge"]
    thr = bankfull if bankfull is not None else q.quantile(0.98)
    df["bankfull_threshold"] = thr
    df["discharge_norm"] = q / thr if thr and not pd.isna(thr) else np.nan
    df["discharge_chg_1d"] = q - q.shift(1)
    df["discharge_chg_3d"] = q - q.shift(3)
    df["discharge_rise_rate"] = df["discharge_chg_1d"] / q.shift(1).replace(0, np.nan)
    df["discharge_7d_mean"] = q.rolling(7, min_periods=1).mean()
    df["discharge_pctile"] = q.rank(pct=True)

    # ---- upstream catchment rainfall -----------------------------------------------
    if up_weather is not None:
        u = up_weather[["date", "precipitation_sum"]].rename(
            columns={"precipitation_sum": "upstream_precip"})
        df = df.merge(u, on="date", how="left")
        ur = df["upstream_precip"]
        df["upstream_precip_3d"] = ur.rolling(3, min_periods=1).sum()
        df["upstream_precip_7d"] = ur.rolling(7, min_periods=1).sum()
    else:
        for c in UPSTREAM_COLS:
            df[c] = np.nan

    # ---- dam release proxy ----------------------------------------------------------
    if up_discharge is not None:
        u = up_discharge[["date", "river_discharge"]].rename(
            columns={"river_discharge": "dam_outflow"})
        df = df.merge(u, on="date", how="left")
        dq = pd.to_numeric(df["dam_outflow"], errors="coerce")
        df["dam_outflow"] = dq
        dthr = dam_bankfull if dam_bankfull is not None else dq.quantile(0.98)
        df["dam_outflow_norm"] = dq / dthr if dthr and not pd.isna(dthr) else np.nan
        df["dam_outflow_chg_1d"] = dq - dq.shift(1)
        df["dam_outflow_chg_3d"] = dq - dq.shift(3)
        df["has_upstream_dam"] = 1
    else:
        for c in DAM_COLS:
            df[c] = np.nan
        df["has_upstream_dam"] = 0

    # ---- seasonality -----------------------------------------------------------------
    doy = pd.to_datetime(df["date"]).dt.dayofyear
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["is_monsoon"] = df["month"].between(6, 10).astype(int)
    return df


def add_label(df, bankfull, lead_days=3):
    """flood_next3 = discharge breaches `bankfull` on any of the next `lead_days` days."""
    fwd = pd.concat([df["discharge"].shift(-k) for k in range(1, lead_days + 1)], axis=1)
    lab = (fwd.max(axis=1) >= bankfull).astype("float")
    lab[fwd.isna().all(axis=1)] = np.nan
    df["flood_next3"] = lab
    return df
