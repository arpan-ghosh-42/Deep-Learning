"""
Train the hydro-meteorological branch: gradient boosting on weather + river state.

Task: given today's observed weather and river level, tomorrow's forecast weather, the
catchment's recent wetness and the upstream dam's outflow, will this gauge breach its
bankfull threshold within the next 3 days?

Split is TEMPORAL, never random: training on 2005-2018, validating on 2019-2022 and
testing on 2023-2026. A random split of daily river data leaks badly - tomorrow's row is
almost the same as today's - and would report a fake 0.99 AUC.

Two baselines are reported alongside the model so the numbers mean something:
  persistence  - "the river will flood iff it is already above threshold today"
  rain-only    - gradient boosting on rainfall features alone

Usage:
    python src/train_tabular.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

TRAIN_END = "2018-12-31"
VAL_END = "2022-12-31"

DROP = {"flood_next3", "date", "station_id", "station_name", "river", "district",
        "bankfull_threshold"}

RAIN_ONLY = ["precipitation_sum", "precipitation_sum_t1", "precip_2day_total",
             "precip_3d", "precip_7d", "precip_15d", "precip_30d", "api_15d",
             "rain_days_7d", "upstream_precip", "upstream_precip_3d", "upstream_precip_7d"]


def metrics(y, p, probabilistic=True):
    """ROC-AUC and PR-AUC are ranking metrics and accept any score. Brier needs a
    calibrated probability, so it is only reported for models that emit one - the
    persistence baseline is a normalised river level, not a probability."""
    out = dict(
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        base_rate=float(np.mean(y)),
    )
    if probabilistic:
        out["brier"] = float(brier_score_loss(y, p))
    return out


def best_f1(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    i = int(np.nanargmax(f1))
    return float(f1[i]), float(thr[min(i, len(thr) - 1)]), float(prec[i]), float(rec[i])


def main():
    df = pd.read_csv(C.PROC / "flood_tabular.csv", parse_dates=["date"])
    df = df.sort_values(["date", "station_id"]).reset_index(drop=True)

    feats = [c for c in df.columns
             if c not in DROP and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].to_numpy(dtype=np.float32)
    y = df["flood_next3"].to_numpy(dtype=int)

    tr = df.date <= TRAIN_END
    va = (df.date > TRAIN_END) & (df.date <= VAL_END)
    te = df.date > VAL_END

    print(f"rows {len(df):,}   features {len(feats)}")
    for n, m in (("train", tr), ("val", va), ("test", te)):
        print(f"  {n:<6} {m.sum():>7,} rows  "
              f"{df.loc[m,'date'].min().date()} -> {df.loc[m,'date'].max().date()}  "
              f"flood rate {y[m.to_numpy()].mean()*100:5.2f}%")

    clf = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=0, class_weight="balanced")
    print("\nfitting main model ...")
    clf.fit(X[tr.to_numpy()], y[tr.to_numpy()])

    out = {"n_features": len(feats), "features": feats,
           "train_end": TRAIN_END, "val_end": VAL_END, "n_rows": int(len(df))}
    for name, m in (("val", va), ("test", te)):
        p = clf.predict_proba(X[m.to_numpy()])[:, 1]
        out[name] = metrics(y[m.to_numpy()], p)
        f1, thr, pr, rc = best_f1(y[m.to_numpy()], p)
        out[name].update(best_f1=f1, best_threshold=thr, precision_at_best=pr,
                         recall_at_best=rc)
        print(f"\n{name.upper()}  ROC-AUC {out[name]['roc_auc']:.4f}   "
              f"PR-AUC {out[name]['pr_auc']:.4f}  (base rate {out[name]['base_rate']:.4f})")
        print(f"       best F1 {f1:.3f} at p>{thr:.3f}  "
              f"(precision {pr:.3f}, recall {rc:.3f})   Brier {out[name]['brier']:.5f}")

    # ---- baselines -------------------------------------------------------------
    ten = te.to_numpy()
    persist = df.loc[te, "discharge_norm"].to_numpy(dtype=float)
    out["baseline_persistence"] = metrics(y[ten], np.nan_to_num(persist),
                                          probabilistic=False)
    print(f"\nbaseline persistence (today's normalised level)   "
          f"ROC-AUC {out['baseline_persistence']['roc_auc']:.4f}  "
          f"PR-AUC {out['baseline_persistence']['pr_auc']:.4f}")

    rain_cols = [f for f in RAIN_ONLY if f in feats]
    ri = [feats.index(c) for c in rain_cols]
    rclf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                          random_state=0, class_weight="balanced")
    rclf.fit(X[tr.to_numpy()][:, ri], y[tr.to_numpy()])
    pr_rain = rclf.predict_proba(X[ten][:, ri])[:, 1]
    out["baseline_rain_only"] = metrics(y[ten], pr_rain)
    print(f"baseline rain-only ({len(rain_cols)} features)                "
          f"ROC-AUC {out['baseline_rain_only']['roc_auc']:.4f}  "
          f"PR-AUC {out['baseline_rain_only']['pr_auc']:.4f}")

    # ---- permutation importance on the validation window -----------------------
    print("\ncomputing permutation importance (val, 6 repeats) ...")
    sub = np.random.default_rng(0).choice(np.flatnonzero(va.to_numpy()),
                                          size=min(20000, int(va.sum())), replace=False)
    imp = permutation_importance(clf, X[sub], y[sub], n_repeats=6,
                                 scoring="average_precision", random_state=0, n_jobs=1)
    order = np.argsort(imp.importances_mean)[::-1]
    top = [(feats[i], round(float(imp.importances_mean[i]), 5)) for i in order[:25]]
    out["top_features"] = top
    print("\nTop 20 features by permutation importance (drop in PR-AUC):")
    for n, v in top[:20]:
        print(f"  {n:<34} {v:.5f}")

    # ---- per-station test performance ------------------------------------------
    p_te = clf.predict_proba(X[ten])[:, 1]
    rows = []
    for sid, g in df.loc[te].assign(p=p_te).groupby("station_id"):
        if g.flood_next3.nunique() < 2:
            continue
        rows.append(dict(station_id=sid, station=g.station_name.iloc[0],
                         river=g.river.iloc[0], n=len(g),
                         flood_rate=round(g.flood_next3.mean(), 4),
                         roc_auc=round(roc_auc_score(g.flood_next3, g.p), 4),
                         pr_auc=round(average_precision_score(g.flood_next3, g.p), 4)))
    per_st = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    per_st.to_csv(C.REPORTS / "tabular_per_station.csv", index=False)
    print(f"\nPer-station test ROC-AUC (median {per_st.roc_auc.median():.3f}):")
    print(per_st.to_string(index=False))

    import pickle
    with open(C.MODELS / "tabular_gbm.pkl", "wb") as f:
        pickle.dump({"model": clf, "features": feats,
                     "threshold": out["test"]["best_threshold"]}, f)
    json.dump(out, open(C.REPORTS / "tabular_metrics.json", "w"), indent=2)
    print(f"\nSaved {C.MODELS / 'tabular_gbm.pkl'} and {C.REPORTS / 'tabular_metrics.json'}")


if __name__ == "__main__":
    main()
