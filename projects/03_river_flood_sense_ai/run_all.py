"""
Reproduce the whole project from an empty datasets/ folder.

    python run_all.py              # everything
    python run_all.py --from train_cnn_ms      # resume at a step
    python run_all.py --list

Downloads are resumable: every step skips work that is already on disk, so re-running
after an interrupted network session is safe and cheap.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("download_discharge", ["src/download_discharge.py"],
     "GloFAS river discharge for 28 gauges + dam outlets (2005-2026)"),
    ("download_weather", ["src/download_weather.py"],
     "ERA5 daily weather, 20 variables per gauge"),
    ("download_satellite", ["src/download_satellite.py", "--workers", "12"],
     "446 hand-labelled Sen1Floods11 chips (~1.0 GB)"),
    ("build_tabular", ["src/build_tabular.py"],
     "feature engineering + flood labels -> flood_tabular.csv"),
    ("build_images", ["src/build_images.py"],
     "GeoTIFF -> training arrays + preview PNGs"),
    ("train_cnn_ms", ["src/train_cnn.py", "--epochs", "60", "--channels", "ms"],
     "U-Net water segmentation, multispectral"),
    ("train_cnn_rgb", ["src/train_cnn.py", "--epochs", "60", "--channels", "rgb"],
     "U-Net water segmentation, RGB-only (what the app uses for uploads)"),
    ("train_tabular", ["src/train_tabular.py"],
     "gradient boosting flood forecaster + baselines"),
    ("build_multimodal", ["src/build_multimodal.py"],
     "pair every chip with the weather that produced it"),
    ("train_fusion", ["src/train_fusion.py"],
     "fusion head: image-only vs hydro-only vs fused"),
    ("make_report", ["src/make_report.py"],
     "figures into reports/"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n, _, d in STEPS:
            print(f"  {n:<22} {d}")
        return

    steps = STEPS
    if args.only:
        steps = [s for s in STEPS if s[0] == args.only]
    elif args.start:
        names = [s[0] for s in STEPS]
        if args.start not in names:
            raise SystemExit(f"unknown step {args.start!r}; try --list")
        steps = STEPS[names.index(args.start):]

    for i, (name, cmd, desc) in enumerate(steps, 1):
        print(f"\n{'=' * 78}\n[{i}/{len(steps)}] {name} — {desc}\n{'=' * 78}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-u", *cmd], cwd=ROOT)
        if r.returncode != 0:
            raise SystemExit(f"\nstep {name!r} failed (exit {r.returncode}). "
                             f"Fix it, then: python run_all.py --from {name}")
        print(f"-- {name} done in {time.time() - t0:.0f}s")

    print("\nAll steps complete.  Launch the demo with:\n"
          "    streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
