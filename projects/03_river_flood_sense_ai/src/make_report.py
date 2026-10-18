"""
Generate the figures that make the project legible at a glance.

    python src/make_report.py

Writes PNGs into reports/.
"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C               # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, ORANGE, RED, GREY = "#1f77b4", "#ff7f0e", "#d62728", "#888888"


def fig_station_map():
    snap = pd.read_csv(C.RAW_DISCHARGE / "_snap_metadata.csv")
    tab = pd.read_csv(C.PROC / "flood_tabular.csv", usecols=["station_id", "flood_next3"])
    rate = tab.groupby("station_id").flood_next3.mean()

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    g = snap[snap.kind == "gauge"].copy()
    d = snap[snap.kind == "dam"]
    g["rate"] = g.id.map(rate).fillna(0)
    sc = ax.scatter(g.snap_lon, g.snap_lat, s=40 + 260 * np.sqrt(g.monsoon_mean_m3s /
                    g.monsoon_mean_m3s.max()), c=g.rate * 100, cmap="YlOrRd",
                    edgecolor="k", linewidth=.5, zorder=3, vmin=0)
    ax.scatter(d.snap_lon, d.snap_lat, marker="^", s=90, c="#2c7fb8",
               edgecolor="k", linewidth=.5, zorder=4, label="dam / barrage")
    for r in snap.itertuples():
        ax.annotate(r.name, (r.snap_lon, r.snap_lat), fontsize=6.2,
                    xytext=(4, 3), textcoords="offset points", color="#333")
    plt.colorbar(sc, ax=ax, label="days flagged as pre-flood (%)", shrink=.8)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("West Bengal gauge network\nmarker size = monsoon mean discharge")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(C.REPORTS / "fig1_station_map.png"); plt.close(fig)


def fig_hydrograph(station="ghatal_silabati", year=2024):
    df = pd.read_csv(C.PROC / "flood_tabular.csv", parse_dates=["date"])
    s = df[(df.station_id == station) & (df.date.dt.year == year)]
    if s.empty:
        return
    thr = s.bankfull_threshold.iloc[0]
    name = s.station_name.iloc[0]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 5.4), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2, 1]))
    a1.plot(s.date, s.discharge, color=BLUE, lw=1.3, label="river discharge")
    a1.axhline(thr, color=RED, ls="--", lw=1.2,
               label=f"bankfull ({thr:,.0f} m³/s, 98th pct)")
    fl = s[s.flood_next3 == 1]
    a1.scatter(fl.date, fl.discharge, s=9, color=RED, zorder=3,
               label=f"labelled pre-flood ({len(fl)} days)")
    a1.set_ylabel("m³/s"); a1.legend(fontsize=8, loc="upper left")
    a1.set_title(f"{name} — {s.river.iloc[0]}, {year}")

    a2.bar(s.date, s.precipitation_sum, color="#4c9be8", width=1.0, label="daily rain")
    a2.set_ylabel("mm"); a2.legend(fontsize=8, loc="upper left")
    a2.set_xlabel("date")
    fig.tight_layout(); fig.savefig(C.REPORTS / f"fig2_hydrograph_{station}_{year}.png")
    plt.close(fig)


def fig_importance():
    p = C.REPORTS / "tabular_metrics.json"
    if not p.exists():
        return
    m = json.load(open(p))
    top = m.get("top_features", [])[:18][::-1]
    if not top:
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.barh([t[0] for t in top], [t[1] for t in top], color=BLUE)
    ax.set_xlabel("drop in PR-AUC when the feature is shuffled")
    ax.set_title("What the flood model actually uses")
    fig.tight_layout(); fig.savefig(C.REPORTS / "fig3_feature_importance.png"); plt.close(fig)


def fig_pr_curve():
    from sklearn.metrics import precision_recall_curve, roc_curve
    mp = C.MODELS / "tabular_gbm.pkl"
    if not mp.exists():
        return
    b = pickle.load(open(mp, "rb"))
    df = pd.read_csv(C.PROC / "flood_tabular.csv", parse_dates=["date"])
    meta = json.load(open(C.REPORTS / "tabular_metrics.json"))
    te = df.date > meta["val_end"]
    X = df.loc[te, b["features"]].to_numpy(np.float32)
    y = df.loc[te, "flood_next3"].to_numpy(int)
    p = b["model"].predict_proba(X)[:, 1]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.0))
    pr, rc, _ = precision_recall_curve(y, p)
    a1.plot(rc, pr, color=BLUE, lw=1.8)
    a1.axhline(y.mean(), color=GREY, ls=":", label=f"base rate {y.mean():.3f}")
    a1.set_xlabel("recall"); a1.set_ylabel("precision")
    a1.set_title(f"Precision–recall (test)\nPR-AUC {meta['test']['pr_auc']:.3f}")
    a1.legend(fontsize=8)

    fpr, tpr, _ = roc_curve(y, p)
    a2.plot(fpr, tpr, color=ORANGE, lw=1.8)
    a2.plot([0, 1], [0, 1], color=GREY, ls=":")
    a2.set_xlabel("false positive rate"); a2.set_ylabel("true positive rate")
    a2.set_title(f"ROC (test)\nAUC {meta['test']['roc_auc']:.3f}")
    fig.tight_layout(); fig.savefig(C.REPORTS / "fig4_tabular_curves.png"); plt.close(fig)


def fig_segmentation(n=5, channels=None):
    import torch
    from train_cnn import UNet, spectral_indices
    ck_path = None
    for c in (channels, "ms", "rgb"):
        if c and (C.MODELS / f"unet_water_{c}.pt").exists():
            ck_path, channels = C.MODELS / f"unet_water_{c}.pt", c
            break
    if ck_path is None:
        return
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = UNet(cin=ck["in_ch"], base=ck["base"]); model.load_state_dict(ck["state_dict"])
    model.eval()

    meta = pd.read_csv(C.PROC / "chip_meta.csv")
    Y = np.load(C.PROC_IMG / "Y.npy")
    rgb8 = np.load(C.PROC_IMG / "RGB8.npy")
    X = (rgb8.astype(np.float32) / 255.0 if channels == "rgb"
         else np.load(C.PROC_IMG / "X.npy").astype(np.float32))
    xt = torch.from_numpy(X)
    if channels == "ms":
        xt = spectral_indices(xt)

    te = meta.index[(meta.split == "test") & (meta.event == "India")].tolist()
    te = (te or meta.index[meta.split == "test"].tolist())[:n]
    if not te:
        return

    fig, axes = plt.subplots(len(te), 3, figsize=(7.2, 2.45 * len(te)))
    axes = np.atleast_2d(axes)
    with torch.no_grad():
        prob = torch.sigmoid(model(xt[te])).numpy()
    for r, i in enumerate(te):
        img = rgb8[i].transpose(1, 2, 0)
        truth = Y[i]
        axes[r, 0].imshow(img)
        axes[r, 0].set_ylabel(meta.chip[i], fontsize=6.5)
        axes[r, 1].imshow(np.ma.masked_where(truth == 255, truth), cmap="Blues", vmin=0, vmax=1)
        axes[r, 2].imshow(prob[r] > 0.5, cmap="Blues", vmin=0, vmax=1)
        iou = (((prob[r] > .5) & (truth == 1)).sum() /
               max(((prob[r] > .5) | (truth == 1)).sum(), 1))
        axes[r, 2].set_xlabel(f"IoU {iou:.2f}", fontsize=7)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([]); axes[r, c].grid(False)
    for c, t in enumerate(["Sentinel-2 RGB", "hand-labelled water", f"U-Net ({channels})"]):
        axes[0, c].set_title(t, fontsize=9)
    fig.suptitle("Water segmentation on held-out Ganga-basin chips", fontsize=10)
    fig.tight_layout(); fig.savefig(C.REPORTS / "fig5_segmentation.png"); plt.close(fig)


def fig_fusion():
    p = C.REPORTS / "fusion_metrics.json"
    if not p.exists():
        return
    m = json.load(open(p))
    keys = [k for k in ("image-only", "hydro-only", "FUSED") if k in m]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = np.arange(len(keys)); w = 0.38
    ax.bar(x - w/2, [m[k]["roc_auc"] for k in keys], w, label="ROC-AUC", color=BLUE)
    ax.bar(x + w/2, [m[k]["pr_auc"] for k in keys], w, label="PR-AUC", color=ORANGE)
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(keys); ax.set_ylim(0, 1.05)
    ax.set_title("Does fusing imagery with weather help?")
    ax.legend(fontsize=8)
    for i, k in enumerate(keys):
        ax.text(i - w/2, m[k]["roc_auc"] + .02, f"{m[k]['roc_auc']:.2f}",
                ha="center", fontsize=7.5)
        ax.text(i + w/2, m[k]["pr_auc"] + .02, f"{m[k]['pr_auc']:.2f}",
                ha="center", fontsize=7.5)
    fig.tight_layout(); fig.savefig(C.REPORTS / "fig6_fusion.png"); plt.close(fig)


def fig_cnn_history():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    any_ = False
    for ch, col in (("ms", BLUE), ("rgb", ORANGE)):
        p = C.REPORTS / f"cnn_metrics_{ch}.json"
        if not p.exists():
            continue
        h = json.load(open(p))
        ax.plot([e["epoch"] for e in h["history"]], [e["val_iou"] for e in h["history"]],
                color=col, lw=1.5, label=f"{ch} (test IoU {h['test_iou']:.3f})")
        any_ = True
    if not any_:
        plt.close(fig); return
    ax.set_xlabel("epoch"); ax.set_ylabel("validation water IoU")
    ax.set_title("U-Net training: multispectral vs RGB-only")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.REPORTS / "fig7_cnn_training.png"); plt.close(fig)


if __name__ == "__main__":
    made = []
    for fn in (fig_station_map, fig_hydrograph, fig_importance, fig_pr_curve,
               fig_segmentation, fig_fusion, fig_cnn_history):
        try:
            fn()
            made.append(fn.__name__)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {fn.__name__}: {type(e).__name__}: {str(e)[:130]}")
    print("figures written:", ", ".join(made))
    for f in sorted(C.REPORTS.glob("fig*.png")):
        print(f"  {f.name}  ({f.stat().st_size/1e3:.0f} kB)")
