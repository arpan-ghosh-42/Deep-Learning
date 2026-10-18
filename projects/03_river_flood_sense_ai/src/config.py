"""
River Flood Sense AI - central configuration.

Paths, the West Bengal station network, date ranges and variable lists live here so the
download / build / train / predict scripts all stay in sync.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
RAW = DATA / "raw"
PROC = DATA / "processed"
RAW_WEATHER = RAW / "weather"
RAW_DISCHARGE = RAW / "discharge"
RAW_SAT = RAW / "satellite"
PROC_IMG = PROC / "images"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

for _p in (RAW_WEATHER, RAW_DISCHARGE, RAW_SAT, PROC, PROC_IMG, MODELS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------
# Period of record. ERA5 reanalysis (Open-Meteo archive) lags real time by ~5 days.
# ---------------------------------------------------------------------------------
START_DATE = "2005-01-01"
END_DATE = "2026-08-20"

# ---------------------------------------------------------------------------------
# Station network.
#
# GAUGES = flood-prone river points across West Bengal. Each carries an `upstream`
#          key naming the reservoir / barrage that regulates it (None = unregulated).
# DAMS   = reservoir and barrage outlets. GloFAS discharge at a dam outlet cell is
#          used as an open PROXY for controlled release, because DVC / Farakka cusec
#          release figures are not published through any open API (see README).
#
# Coordinates are snapped to the best GloFAS river cell at download time.
# ---------------------------------------------------------------------------------
GAUGES = [
    dict(id="farakka_ganga",      name="Farakka",       river="Ganga",        district="Murshidabad",       lat=24.803, lon=87.930, upstream=None),
    dict(id="berhampore_bhagi",   name="Berhampore",    river="Bhagirathi",   district="Murshidabad",       lat=24.100, lon=88.250, upstream="farakka_barrage"),
    dict(id="katwa_bhagirathi",   name="Katwa",         river="Bhagirathi",   district="Purba Bardhaman",   lat=23.650, lon=88.130, upstream="farakka_barrage"),
    dict(id="nabadwip_bhagi",     name="Nabadwip",      river="Bhagirathi",   district="Nadia",             lat=23.408, lon=88.368, upstream="farakka_barrage"),
    dict(id="kolkata_hooghly",    name="Kolkata",       river="Hooghly",      district="Kolkata",           lat=22.575, lon=88.345, upstream="farakka_barrage"),
    dict(id="burdwan_damodar",    name="Bardhaman",     river="Damodar",      district="Purba Bardhaman",   lat=23.235, lon=87.855, upstream="durgapur_barrage"),
    dict(id="udaynarayanpur",     name="Udaynarayanpur",river="Damodar",      district="Howrah",            lat=22.730, lon=87.990, upstream="durgapur_barrage"),
    dict(id="asansol_barakar",    name="Asansol",       river="Barakar",      district="Paschim Bardhaman", lat=23.680, lon=86.940, upstream="maithon_dam"),
    dict(id="ilambazar_ajay",     name="Ilambazar",     river="Ajay",         district="Birbhum",           lat=23.600, lon=87.550, upstream=None),
    dict(id="ghatal_silabati",    name="Ghatal",        river="Shilabati",    district="Paschim Medinipur", lat=22.665, lon=87.727, upstream="mukutmanipur_dam"),
    dict(id="midnapore_kangsa",   name="Midnapore",     river="Kangsabati",   district="Paschim Medinipur", lat=22.425, lon=87.320, upstream="mukutmanipur_dam"),
    dict(id="kolaghat_rupnaraya", name="Kolaghat",      river="Rupnarayan",   district="Purba Medinipur",   lat=22.432, lon=87.862, upstream="mukutmanipur_dam"),
    dict(id="bankura_dwarakesw",  name="Bankura",       river="Dwarakeswar",  district="Bankura",           lat=23.240, lon=87.070, upstream=None),
    dict(id="gopiballavpur_sub",  name="Gopiballavpur", river="Subarnarekha", district="Jhargram",          lat=22.205, lon=86.900, upstream=None),
    dict(id="krishnanagar_jalan", name="Krishnanagar",  river="Jalangi",      district="Nadia",             lat=23.400, lon=88.500, upstream="farakka_barrage"),
    dict(id="malda_mahananda",    name="Malda",         river="Mahananda",    district="Malda",             lat=25.000, lon=88.140, upstream=None),
    dict(id="siliguri_mahananda", name="Siliguri",      river="Mahananda",    district="Darjeeling",        lat=26.710, lon=88.430, upstream=None),
    dict(id="jalpaiguri_teesta",  name="Jalpaiguri",    river="Teesta",       district="Jalpaiguri",        lat=26.520, lon=88.720, upstream="gajoldoba_barrage"),
    dict(id="coochbehar_torsa",   name="Cooch Behar",   river="Torsa",        district="Cooch Behar",       lat=26.320, lon=89.450, upstream=None),
    dict(id="jaldhaka_dhupguri",  name="Dhupguri",      river="Jaldhaka",     district="Jalpaiguri",        lat=26.600, lon=89.030, upstream=None),
]

DAMS = [
    dict(id="maithon_dam",       name="Maithon Dam",       river="Barakar",    lat=23.783, lon=86.815),
    dict(id="panchet_dam",       name="Panchet Dam",       river="Damodar",    lat=23.663, lon=86.735),
    dict(id="konar_dam",         name="Konar Dam",         river="Konar",      lat=23.860, lon=85.830),
    dict(id="tilaiya_dam",       name="Tilaiya Dam",       river="Barakar",    lat=24.330, lon=85.530),
    dict(id="durgapur_barrage",  name="Durgapur Barrage",  river="Damodar",    lat=23.478, lon=87.297),
    dict(id="mukutmanipur_dam",  name="Mukutmanipur Dam",  river="Kangsabati", lat=22.947, lon=86.788),
    dict(id="farakka_barrage",   name="Farakka Barrage",   river="Ganga",      lat=24.803, lon=87.930),
    dict(id="gajoldoba_barrage", name="Gajoldoba Barrage", river="Teesta",     lat=26.775, lon=88.545),
]

ALL_POINTS = {p["id"]: p for p in GAUGES + DAMS}

# ---------------------------------------------------------------------------------
# Open-Meteo variables and endpoints
# ---------------------------------------------------------------------------------
DAILY_WEATHER_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max",
    "precipitation_sum", "rain_sum", "precipitation_hours",
    "relative_humidity_2m_max", "relative_humidity_2m_mean", "relative_humidity_2m_min",
    "dew_point_2m_mean", "surface_pressure_mean", "pressure_msl_mean",
    "wind_speed_10m_max", "cloud_cover_mean", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "soil_moisture_0_to_7cm_mean", "soil_moisture_0_to_100cm_mean",
    "vapour_pressure_deficit_max",
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# ---------------------------------------------------------------------------------
# Flood labelling
#
# A "flood day" at a gauge = GloFAS river discharge exceeds that station's bankfull
# proxy (a high quantile of its own 2005-2026 daily record). The model target is
# whether that threshold is breached within the next LEAD_DAYS days.
# ---------------------------------------------------------------------------------
BANKFULL_Q = 0.98
LEAD_DAYS = 3

# ---------------------------------------------------------------------------------
# Satellite imagery (Sen1Floods11, hand-labelled subset)
# ---------------------------------------------------------------------------------
GCS_BUCKET = "sen1floods11"
GCS_PREFIX = "v1.1/data/flood_events/HandLabeled"
SAT_LAYERS = ["S2Hand", "LabelHand", "JRCWaterHand"]

# Monsoon-Asia events first: closest hydrological analogue to the Ganga basin.
EVENT_PRIORITY = ["India", "Mekong", "Pakistan", "Sri-Lanka", "Bolivia",
                  "Ghana", "Nigeria", "Somalia", "Paraguay", "Spain", "USA"]

# Sentinel-2 L1C 13-band order: B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12
S2_BAND_IDX = dict(blue=1, green=2, red=3, nir=7, swir1=11)
CHIP_SIZE = 256
FLOOD_FRAC_THRESHOLD = 0.05
