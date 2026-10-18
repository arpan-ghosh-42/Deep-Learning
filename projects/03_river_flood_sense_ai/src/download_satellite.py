"""
Download the hand-labelled Sen1Floods11 flood chips (Sentinel-2 optical + water masks).

Source: gs://sen1floods11 (Cloud to Street / NASA IMPACT, CC BY 4.0). The bucket is public,
so we read it over plain HTTPS via the GCS JSON API - no gsutil and no credentials needed.

Three layers are pulled per chip:
  S2Hand       - Sentinel-2 L1C, 13 bands, 512x512 at 10 m  (the "satellite image")
  LabelHand    - hand-drawn water mask: -1 no-data, 0 land, 1 water  (the training target)
  JRCWaterHand - JRC permanent-water baseline: the river's NORMAL extent

The JRC layer is what makes flood detection possible from a single image: flood water is
observed water MINUS the river's permanent extent, so a wide-but-normal river is not
mistaken for a flood.

Chips are fetched in monsoon-Asia-first order (config.EVENT_PRIORITY) so the most
Ganga-like imagery lands even if the run is cut short.

Usage:
    python src/download_satellite.py              # all 446 hand-labelled chips (~1.1 GB)
    python src/download_satellite.py --max 250    # cap the number of chips
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from net import download_file, gcs_list

BASE = f"https://storage.googleapis.com/{C.GCS_BUCKET}"


def build_index():
    """Return {event: [chip_id, ...]} for chips that have all three layers."""
    per_layer = {}
    for layer in C.SAT_LAYERS:
        items, _ = gcs_list(C.GCS_BUCKET, f"{C.GCS_PREFIX}/{layer}/")
        ids = {}
        for name, size in items:
            fn = name.split("/")[-1]
            if not fn.endswith(".tif"):
                continue
            event, chip, _ = fn.split("_", 2)
            ids[(event, chip)] = size
        per_layer[layer] = ids
        print(f"  {layer:<14} {len(ids)} chips, {sum(ids.values())/1e6:7.1f} MB")

    common = set.intersection(*(set(v) for v in per_layer.values()))
    by_event = defaultdict(list)
    for event, chip in sorted(common):
        by_event[event].append(chip)
    return by_event, per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0, help="max chips to download (0 = all)")
    ap.add_argument("--workers", type=int, default=10, help="parallel download workers")
    args = ap.parse_args()

    print("Indexing gs://sen1floods11 hand-labelled chips ...")
    by_event, per_layer = build_index()
    total = sum(len(v) for v in by_event.values())
    print(f"\n{total} chips with all 3 layers, across {len(by_event)} flood events:")
    for ev in sorted(by_event, key=lambda e: C.EVENT_PRIORITY.index(e) if e in C.EVENT_PRIORITY else 99):
        print(f"  {ev:<12} {len(by_event[ev]):4d}")

    # Monsoon-Asia first, then everything else.
    order = []
    for ev in C.EVENT_PRIORITY:
        order += [(ev, c) for c in by_event.get(ev, [])]
    for ev in sorted(by_event):
        if ev not in C.EVENT_PRIORITY:
            order += [(ev, c) for c in by_event[ev]]
    if args.max:
        order = order[:args.max]

    print(f"\nDownloading {len(order)} chips x {len(C.SAT_LAYERS)} layers "
          f"with {args.workers} parallel workers ...")

    jobs = []
    for event, chip in order:
        for layer in C.SAT_LAYERS:
            fn = f"{event}_{chip}_{layer}.tif"
            jobs.append((f"{BASE}/{C.GCS_PREFIX}/{layer}/{fn}", C.RAW_SAT / layer / fn))

    counts = {"new": 0, "cached": 0, "fail": 0}
    lock = Lock()
    done = 0
    t0 = time.time()

    def work(job):
        url, dest = job
        try:
            return "new" if download_file(url, dest) else "cached"
        except Exception as e:  # noqa: BLE001
            print(f"    FAILED {dest.name}: {str(e)[:90]}")
            return "fail"

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(work, jobs):
            with lock:
                counts[res] += 1
                done += 1
                if done % 90 == 0 or done == len(jobs):
                    mb = sum(f.stat().st_size for f in C.RAW_SAT.rglob("*.tif")) / 1e6
                    rate = done / max(time.time() - t0, 1)
                    eta = (len(jobs) - done) / max(rate, 1e-6) / 60
                    print(f"  [{done:5d}/{len(jobs)}] new={counts['new']} "
                          f"cached={counts['cached']} fail={counts['fail']}  "
                          f"{mb:7.1f} MB  {rate:.1f} files/s  ETA {eta:.1f} min")

    rows = []
    for event, chip in order:
        paths = {l: f"datasets/raw/satellite/{l}/{event}_{chip}_{l}.tif" for l in C.SAT_LAYERS}
        if all((C.ROOT / p).exists() for p in paths.values()):
            rows.append(dict(event=event, chip_id=chip, **paths))

    idx = pd.DataFrame(rows)
    idx.to_csv(C.RAW_SAT / "chip_index.csv", index=False)
    print(f"\nWrote chip index ({len(idx)} chips) -> {C.RAW_SAT / 'chip_index.csv'}")


if __name__ == "__main__":
    main()
