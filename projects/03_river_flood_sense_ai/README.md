# 🌊 River Flood Sense AI

Flood-risk assessment for West Bengal's rivers, combining **satellite imagery** with the
**hydro-meteorology that actually causes floods** — rainfall, antecedent catchment
wetness, temperature, humidity, river discharge and upstream dam releases.

Upload a satellite view of a river reach, give it today's and tomorrow's conditions (or
let it fetch them live), and it returns how much of the scene is abnormally under water
and how likely that gauge is to breach bankfull within three days.

> **Demo / portfolio project.** Every dataset is real and downloaded from public APIs, and
> the evaluation is honest — including where it underperforms. It is *not* an operational
> warning system. For real warnings use the
> [WB Irrigation & Waterways Department](https://wbiwd.gov.in/) and IMD bulletins.

---

## What it does

Three questions, answered by three models, because they are genuinely different questions:

| | Question | Model |
|---|---|---|
| **Image branch** | How much of this scene is water *right now*, and how much of that is more than this river normally carries? | U-Net water segmentation |
| **Forecast branch** | Given the weather, the soaked-or-dry catchment, the river's current level and the upstream dam's outflow — will this gauge breach bankfull within 3 days? | Gradient boosting |
| **Fusion head** | Combining what we can see with what we can measure, is this scene genuinely flooded? | Logistic fusion |

The key idea on the image side: **flood extent = observed water − the river's normal
extent.** A wide river is not a flood; an *unusually* wide river is. The JRC permanent-water
layer supplies that baseline during training.

---

## The data (all downloaded, nothing synthetic)

| Dataset | Source | What we pull |
|---|---|---|
| **River discharge** | [Open-Meteo Flood API](https://open-meteo.com/en/docs/flood-api) → Copernicus **GloFAS v4** reanalysis | Daily discharge, 2005-01-01 → 2026-08-20, at 20 West Bengal gauges + 8 dam/barrage outlets |
| **Weather** | [Open-Meteo Historical API](https://open-meteo.com/en/docs/historical-weather-api) → **ERA5 / ERA5-Land** | 20 daily variables per gauge — temperature, humidity, rainfall, soil moisture, pressure, radiation, evapotranspiration |
| **Satellite imagery** | [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) (Cloud to Street / NASA IMPACT, CC BY 4.0) | 446 hand-labelled 512×512 Sentinel-2 chips across 11 global flood events, with hand-drawn water masks and JRC permanent-water baselines |
| **Live conditions** | Open-Meteo forecast + GloFAS forecast | 40 days of history + 3-day forecast, for real-time scoring in the app |

No API keys, no `gsutil`, no Kaggle credentials — everything is fetched over plain HTTPS
and is resumable.

### The gauge network

20 flood-prone river points, each linked to the reservoir or barrage that regulates it:

- **Ganga / Bhagirathi–Hooghly** — Farakka, Berhampore, Katwa, Nabadwip, Kolkata, Krishnanagar
- **Damodar system** — Bardhaman, Udaynarayanpur, Asansol *(fed by Maithon, Panchet, Durgapur)*
- **Western rivers** — Ghatal (Shilabati), Midnapore (Kangsabati), Kolaghat (Rupnarayan), Bankura (Dwarakeswar), Gopiballavpur (Subarnarekha), Ilambazar (Ajay)
- **North Bengal** — Jalpaiguri (Teesta), Siliguri & Malda (Mahananda), Cooch Behar (Torsa), Dhupguri (Jaldhaka)

**Grid snapping.** GloFAS resolves rivers on a 0.05° grid, so a town's nominal coordinate
often lands on a minor cell — Farakka reads **2 m³/s** on its raw coordinate and
**~30,000 m³/s** one cell away. Every point is snapped by probing a 5×5 block of cells
over a full monsoon and keeping the main channel. Every snap is logged with its distance
and mean flow in `datasets/raw/discharge/_snap_metadata.csv`.

---

## How a flood is defined

A **flood day** at a gauge is when GloFAS discharge exceeds that station's **bankfull
proxy** — the 98th percentile of its *own* 2005–2026 daily record. Per-river thresholds
are essential: 2,000 m³/s is a drought on the Ganga and a catastrophe on the Ajay.

The model's target is `flood_next3` — a breach on **any of the next 3 days**, which makes
it a genuine short-lead forecasting task rather than nowcasting.

---

## Features

~50 features per gauge-day, all knowable on the day of prediction:

- **Today's weather** — 20 ERA5 variables at the gauge
- **Tomorrow's forecast** — max/min temperature, humidity, rainfall, rain hours, cloud, pressure
- **Antecedent wetness** — 3/7/15/30-day rainfall totals, rain-days, an Antecedent
  Precipitation Index (`API_t = 0.9·API_{t−1} + rain_t`), and 7-day soil-moisture memory
- **Upstream catchment rain** — rainfall over the controlling reservoir's catchment
- **River state** — discharge, discharge as a fraction of bankfull, 1- and 3-day rise
  rates, 7-day mean, percentile rank
- **Dam release proxy** — outflow at the controlling dam/barrage and its 1/3-day change
- **Seasonality** — month, day-of-year harmonics, monsoon flag

---

## Results

### Dataset actually built

| | |
|---|---|
| Tabular rows | **157,420** (20 gauges × 7,902 days, 2005-01-31 → 2026-08-19) |
| Features | 57 |
| Flood-day rate | 2.78% |
| Satellite chips | 430 usable of 446 (16 dropped as mostly no-data) |
| Flooded chips | 136 / 430 (31.6%) |
| Raw data on disk | ~1.2 GB |

### Flood forecaster — will this gauge breach bankfull within 3 days?

Test window **2023-01-01 → 2026-08-19** (26,540 rows, 3.71% flood days), never seen in training.

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| **Gradient boosting (all features)** | **0.9890** | **0.8619** |
| baseline: persistence (today's normalised level) | 0.9684 | 0.7938 |
| baseline: rainfall features only | 0.9144 | 0.2299 |

Best F1 **0.797** at p > 0.884 (precision 0.847, recall 0.752), Brier 0.0213.
Per-station test ROC-AUC ranges 0.956 (Malda) → 0.997 (Farakka), **median 0.991**.

The persistence comparison is the one that matters: a river already above its threshold is
the single strongest predictor, so any honest evaluation has to beat that. The model does,
mostly on PR-AUC (0.862 vs 0.794) — i.e. it is better at the rare, expensive positives.

**Top features by permutation importance** (drop in PR-AUC when shuffled):

```
discharge_pctile        0.599     <- current river state dominates, as it should
precip_30d              0.066     <- a month of accumulated catchment wetness
cloud_cover_mean_t1     0.034     <- tomorrow's cloud = tomorrow's rain
discharge_norm          0.025
api_15d                 0.019     <- antecedent precipitation index
precipitation_sum_t1    0.019     <- tomorrow's forecast rainfall
discharge_rise_rate     0.011
```

### Label sanity-check against real West Bengal floods

The labels are derived purely from discharge thresholds, with no knowledge of news events —
yet they independently light up on the right dates at the right stations:

| Episode | Stations flagged |
|---|---|
| **Sep–Oct 2024** (DVC release + Bay depression) | **16 of 20** |
| Aug 2017 (north Bengal + Malda) | 10 — Cooch Behar, Jalpaiguri, Siliguri, Malda, Dhupguri, Katwa, Nabadwip, Berhampore, Krishnanagar, Farakka |
| Oct 2021 (post-monsoon depression) | 10 — Damodar/south Bengal cluster |
| Aug 2021 (Damodar basin) | 5 |

The Aug 2017 set is exactly the north Bengal districts that actually flooded, and the Oct 2021
set is exactly the southern Damodar cluster. That geographic separation is not something a
threshold could fake.

### Satellite water segmentation

| Branch | Input | Test water IoU | Test F1 | Params |
|---|---|---|---|---|
| **`ms`** | R, G, B, NIR, SWIR1, NDWI, MNDWI | **0.7351** | **0.8473** | 1.09 M |
| `rgb` | contrast-stretched R, G, B only | 0.2113 | 0.3488 | 1.93 M |

Trained on 300 chips, CPU-only.

> **Note on the RGB number.** Two recipes were tried: base=24 / lr 3e-3 / 60 epochs reached
> IoU 0.280, and base=32 / lr 8e-4 / 35 epochs (stopped early, evaluated from its best
> checkpoint by `src/eval_cnn.py`) reached 0.211 — the currently shipped weights. Lowering the
> learning rate made it *worse*, so the ceiling here is the input, not the optimiser. The RGB
> branch is a documented fallback, not the headline result; the notebook defaults to `ms`.

Predicted vs hand-labelled **water fraction**:

| | correlation |
|---|---|
| held-out test chips | **0.928** |
| Ganga/Brahmaputra (India event) chips | **0.994** |
| mean absolute error | 0.0132 (1.3 percentage points of chip area) |

**The RGB gap is the interesting result.** Dropping NIR and SWIR costs most of the accuracy,
because in true colour turbid flood water looks much like wet soil — it is the infrared bands
that make water reliably dark. That is exactly why the app/notebook uses the multispectral
branch whenever a real Sentinel-2 product is supplied, and treats an ordinary RGB upload as
the degraded fallback it genuinely is.

### Fusion — and an honest negative result

Target is the **hand-drawn** flood label (observed water minus JRC permanent water), so it
comes from human annotation rather than from either model.

| Branch | Features | ROC-AUC | PR-AUC | Accuracy |
|---|---|---|---|---|
| **image-only** | 3 | **0.9756** | **0.9548** | 0.923 |
| hydro-only | 26 | 0.6467 | 0.4930 | 0.615 |
| FUSED | 29 | 0.9733 | 0.9423 | 0.892 |

**Fusion did not beat the image branch, and that is reported as measured rather than tuned
away.** The reason is structural: every Sen1Floods11 chip comes from a flood event that was
already happening, so regional weather is nearly constant across the chips of one event, and
whether a *particular 5 km tile* is under water comes down to local topography. Weather
predicts *when a river floods* (Section above, AUC 0.989); it does not predict *which tile of
an ongoing flood is wet*. Those are different questions, and the honest read is that the two
branches belong side by side rather than averaged into one number.


---

## Project layout

```
03_river_flood_sense_ai/
├── run_all.py                  one-command reproduction of the whole pipeline
├── requirements.txt
├── src/
│   ├── config.py               station network, date ranges, thresholds
│   ├── net.py                  retrying HTTP with rate-limit backoff
│   ├── features.py             SHARED feature engineering (train == inference)
│   ├── download_discharge.py   GloFAS + grid snapping
│   ├── download_weather.py     ERA5, chunked to stay inside the free quota
│   ├── download_satellite.py   Sen1Floods11 over plain HTTPS, parallel
│   ├── build_tabular.py        features + flood labels
│   ├── build_images.py         GeoTIFF → arrays + preview PNGs
│   ├── build_multimodal.py     pairs each chip with the weather that produced it
│   ├── train_cnn.py            U-Net (multispectral and RGB-only)
│   ├── train_tabular.py        gradient boosting + baselines
│   ├── train_fusion.py         image vs hydro vs fused
│   ├── predict.py              inference + CLI
│   └── make_report.py          figures
├── app/streamlit_app.py        interactive demo
├── datasets/
│   ├── raw/{weather,discharge,satellite}/
│   └── processed/              flood_tabular.csv, chip_meta.csv, multimodal.csv, arrays
├── models/                     unet_water_ms.pt, unet_water_rgb.pt, tabular_gbm.pkl, fusion.pkl
└── reports/                    metrics JSON + figures
```

---

## Running it

```bash
python -m pip install -r requirements.txt

python run_all.py               # download everything + train everything
python run_all.py --list        # see the steps
python run_all.py --from train_tabular    # resume partway

streamlit run app/streamlit_app.py        # the demo
```

Command-line scoring:

```bash
# live conditions for a gauge
python src/predict.py --station ghatal_silabati

# with a satellite image
python src/predict.py --station farakka_ganga \
    --image datasets/processed/images/preview/India_1017769.png

# a what-if scenario: heavy rain plus a big dam release
python src/predict.py --station udaynarayanpur \
    --rain-today 180 --rain-tomorrow 140 --dam-outflow 4000
```

Downloading takes roughly 45–70 minutes on a home connection (Open-Meteo's free tier is
rate-limited by request weight, and the downloaders back off politely). Training is
CPU-only and takes about 30 minutes.

---

## Two design decisions worth explaining

**Why two U-Nets.** The multispectral model gets 7 channels — red, green, blue, NIR,
SWIR1 plus NDWI and MNDWI. Those two indices are what make a ~430-image dataset trainable:
they already encode most of the water signal, so the network only has to learn shape,
context and the cases the indices get wrong (cloud shadow, turbid flood water, wet soil).

But a user uploading "a satellite image of the Ganga" has an ordinary RGB picture — no NIR,
no SWIR. So a second model is trained on **contrast-stretched RGB only**, on byte-identical
data to what the app receives at upload time. Training an RGB model on raw reflectance and
then feeding it stretched PNGs would silently break on every real upload. The app picks the
branch automatically from the file you give it.

**Why the temporal split.** Daily river data is enormously autocorrelated — tomorrow's row
looks almost exactly like today's. A random train/test split leaks across that
autocorrelation and reports a fake ~0.99 AUC. Training is on 2005–2018, validation on
2019–2022 and testing on 2023–2026, so the test window contains flood seasons the model has
never seen — including the September 2024 West Bengal floods.

---

## Data honesty

Things a reviewer should know before trusting a number here:

- **Dam releases are a proxy.** DVC and Farakka publish cusec release figures in press
  bulletins, not through any open API. The "dam outflow" feature is GloFAS discharge at the
  reservoir outlet cell — it tracks release behaviour but is not the operator's logged
  release.
- **GloFAS is a global model.** It represents the Bhagirathi–Hooghly distributaries and the
  tidal lower Damodar poorly, because flow there is split by barrage operation that a global
  model does not know. Those gauges keep genuine rainfall-driven dynamics but understated
  absolute discharge — which is exactly why labels are defined as exceedance of each
  station's *own* high quantile rather than an absolute cumec value.
- **Flood labels are threshold exceedances, not surveyed inundation.** No open
  district-level inundation record exists for West Bengal at daily resolution. Labels are
  cross-checked against known episodes (September 2024, August/October 2021, August 2017)
  in the `build_tabular.py` output.
- **The satellite model is trained on global flood imagery.** Sen1Floods11's Indian event is
  the 2016 Brahmaputra flood in Assam, not West Bengal. It is the closest available
  hand-labelled analogue — same monsoon regime, same turbid braided-river appearance — but
  it is not a West-Bengal-specific training set.
- **Tomorrow's "forecast" is tomorrow's reanalysis during training.** That is an optimistic
  stand-in for a real forecast, so live performance with an actual IMD forecast would be
  somewhat worse than the reported test numbers.
- **The permanent-water baseline is user-supplied at inference.** During training the JRC
  layer provides it; for an arbitrary uploaded image the app exposes it as a slider,
  defaulted to the Ganga-basin median.

---

## Credits

- **Sen1Floods11** — Bonafilia, Tellman, Anderson, Issenberg (2020), *Sen1Floods11: a
  georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1*,
  CVPRW. CC BY 4.0.
- **GloFAS** — Copernicus Emergency Management Service, Global Flood Awareness System.
- **ERA5** — Copernicus Climate Change Service (C3S), via
  [Open-Meteo](https://open-meteo.com/) (CC BY 4.0).
- **JRC Global Surface Water** — Pekel et al. (2016), *Nature*.
