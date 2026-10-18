"""
Train the fusion head, and prove the two branches are actually worth combining.

Target : is_flooded - from the HAND-DRAWN Sen1Floods11 water mask minus the JRC permanent
         water layer. A chip counts as flooded when observed water exceeds the river's
         normal extent by more than 5% of the chip area. The label comes from human
         annotation, not from either model.

Three models are fitted on identical rows and the identical chip split:

  image-only   the U-Net's predicted water fraction
  hydro-only   rainfall, antecedent wetness, soil moisture, temperature, river discharge
  FUSED        both

If fusion does not beat both branches, that is worth knowing and is reported honestly.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

HYDRO = [
    "precipitation_sum", "precipitation_sum_t1", "precip_2day_total",
    "precip_3d", "precip_7d", "precip_15d", "precip_30d", "api_15d", "rain_days_7d",
    "soil_moisture_0_to_7cm_mean", "soil_moisture_0_to_100cm_mean", "soil_moist_7d_mean",
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_max_t1",
    "relative_humidity_2m_mean", "relative_humidity_2m_max", "cloud_cover_mean",
    "pressure_msl_mean", "et0_fao_evapotranspiration",
    "discharge", "discharge_norm", "discharge_chg_1d", "discharge_chg_3d",
    "discharge_7d_mean", "discharge_pctile",
]


def fit_eval(name, cols, df, tr, te, out):
    cols = [c for c in cols if c in df.columns]
    Xtr = df.loc[tr, cols].to_numpy(dtype=float)
    Xte = df.loc[te, cols].to_numpy(dtype=float)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    ytr = df.loc[tr, "is_flooded"].to_numpy(dtype=int)
    yte = df.loc[te, "is_flooded"].to_numpy(dtype=int)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p) if len(set(yte)) > 1 else float("nan")
    ap = average_precision_score(yte, p) if len(set(yte)) > 1 else float("nan")
    acc = ((p > 0.5).astype(int) == yte).mean()
    print(f"  {name:<12} {len(cols):3d} features   ROC-AUC {auc:.4f}   "
          f"PR-AUC {ap:.4f}   acc {acc:.3f}")
    out[name] = dict(n_features=len(cols), features=cols, roc_auc=float(auc),
                     pr_auc=float(ap), accuracy=float(acc))
    return model, cols


def main():
    df = pd.read_csv(C.PROC / "multimodal.csv")

    # Always take the U-Net predictions from chip_meta.csv rather than from whatever was
    # frozen into multimodal.csv. chip_meta is rewritten by every train_cnn.py run, so
    # retraining a CNN does not force a re-download of the multimodal weather join.
    chips = pd.read_csv(C.PROC / "chip_meta.csv")
    pred_cols = [c for c in chips.columns if c.startswith("pred_water_frac")]
    df = df.drop(columns=[c for c in pred_cols if c in df.columns], errors="ignore")
    df = df.merge(chips[["chip", *pred_cols]], on="chip", how="left")

    img_col = next((c for c in ("pred_water_frac_ms", "pred_water_frac_rgb")
                    if c in df.columns), None)
    if img_col is None:
        raise SystemExit("no pred_water_frac_* column - run train_cnn.py first")
    print(f"image branches: {', '.join(pred_cols)}   using: {img_col}")

    # "excess water": the CNN's water fraction above this reach's normal river extent.
    df["excess_water_frac"] = (df[img_col] - df["perm_water_frac"]).clip(lower=0)
    image_feats = [img_col, "excess_water_frac", "perm_water_frac"]

    tr = df.split.isin(["train", "val"]).to_numpy()
    te = (df.split == "test").to_numpy()
    print(f"rows {len(df)}   train+val {tr.sum()}   test {te.sum()}   "
          f"flood rate {df.is_flooded.mean():.3f}   image branch: {img_col}\n")

    out = {"image_column": img_col, "n_rows": int(len(df)),
           "flood_rate": float(df.is_flooded.mean())}
    fit_eval("image-only", image_feats, df, tr, te, out)
    fit_eval("hydro-only", HYDRO, df, tr, te, out)
    model, cols = fit_eval("FUSED", image_feats + HYDRO, df, tr, te, out)

    best = max(("image-only", "hydro-only", "FUSED"),
               key=lambda k: out[k]["roc_auc"] if out[k]["roc_auc"] == out[k]["roc_auc"] else -1)
    print(f"\nbest branch on the held-out chips: {best}")
    if best != "FUSED":
        print("  (fusion did not beat the single branch on this small test split - "
              "reported as measured, not tuned away)")

    med = np.nanmedian(df.loc[tr, cols].to_numpy(dtype=float), axis=0)
    with open(C.MODELS / "fusion.pkl", "wb") as f:
        pickle.dump({"model": model, "features": cols, "image_col": img_col,
                     "impute_median": np.where(np.isnan(med), 0.0, med)}, f)
    json.dump(out, open(C.REPORTS / "fusion_metrics.json", "w"), indent=2)
    print(f"\nSaved {C.MODELS / 'fusion.pkl'} and {C.REPORTS / 'fusion_metrics.json'}")


if __name__ == "__main__":
    main()
