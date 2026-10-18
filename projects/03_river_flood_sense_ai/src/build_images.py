"""
Turn raw Sen1Floods11 GeoTIFF chips into compact training arrays + browsable previews.

For every chip we produce:
  X    (5, 256, 256) float16   red, green, blue, NIR, SWIR1 surface reflectance (0-1)
  Y    (256, 256)    uint8     water mask: 0 land, 1 water, 255 no-data (ignored in loss)
  JRC  (256, 256)    uint8     JRC permanent-water baseline (the river's normal extent)
  RGB8 (3, 256, 256) uint8     contrast-stretched true colour - byte-for-byte what the
                               preview PNG contains, and therefore exactly what the app
                               receives when a user uploads an ordinary satellite image

and a per-chip metadata row with its real-world centroid (read from the GeoTIFF
ModelTiepoint / ModelPixelScale tags) and its flood statistics:

  water_frac  = fraction of the chip that is water on the day of the image
  perm_frac   = fraction that is water even in a normal year (JRC)
  flood_frac  = water_frac - perm_frac, clipped at 0  ->  the actual flood signal
  is_flooded  = flood_frac > FLOOD_FRAC_THRESHOLD

That subtraction is the whole trick: a wide river is not a flood, an unusually wide
river is. The centroid + event date later let us join each image to the weather and
river discharge that produced it (see build_multimodal.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

BANDS = ["red", "green", "blue", "nir", "swir1"]
S2_SCALE = 10000.0   # Sentinel-2 L1C reflectance is stored x10000


def geo_centroid(path):
    """Read the chip's WGS84 centroid from its GeoTIFF tags."""
    with tifffile.TiffFile(path) as t:
        tags = t.pages[0].tags
        sx, sy, _ = tags["ModelPixelScaleTag"].value
        _, _, _, lon_ul, lat_ul, _ = tags["ModelTiepointTag"].value
        h, w = tags["ImageLength"].value, tags["ImageWidth"].value
    return lat_ul - (h / 2) * sy, lon_ul + (w / 2) * sx


def resize_bands(arr, size):
    """arr (C,H,W) float32 -> (C,size,size) with bilinear resampling."""
    out = np.empty((arr.shape[0], size, size), dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = np.asarray(
            Image.fromarray(arr[i]).resize((size, size), Image.BILINEAR), dtype=np.float32)
    return out


def resize_mask(m, size):
    return np.asarray(Image.fromarray(m).resize((size, size), Image.NEAREST))


def stretch_rgb(rgb):
    """Percentile stretch to a viewable true-colour image."""
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        b = rgb[i]
        lo, hi = np.percentile(b[np.isfinite(b)], [2, 98])
        out[i] = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out.transpose(1, 2, 0) * 255).astype(np.uint8)


def main():
    s2_dir = C.RAW_SAT / "S2Hand"
    files = sorted(s2_dir.glob("*_S2Hand.tif"))
    if not files:
        raise SystemExit("No chips found - run src/download_satellite.py first.")

    prev_dir = C.PROC_IMG / "preview"
    prev_dir.mkdir(parents=True, exist_ok=True)

    X, Y, J, RGB, rows = [], [], [], [], []
    skipped = 0
    print(f"Processing {len(files)} chips -> {C.CHIP_SIZE}x{C.CHIP_SIZE}\n")

    for n, f in enumerate(files, 1):
        stem = f.name.replace("_S2Hand.tif", "")
        lab_f = C.RAW_SAT / "LabelHand" / f"{stem}_LabelHand.tif"
        jrc_f = C.RAW_SAT / "JRCWaterHand" / f"{stem}_JRCWaterHand.tif"
        if not (lab_f.exists() and jrc_f.exists()):
            skipped += 1
            continue

        s2 = tifffile.imread(f).astype(np.float32)          # (13,512,512)
        lab = tifffile.imread(lab_f).astype(np.int16)        # -1 / 0 / 1
        jrc = tifffile.imread(jrc_f).astype(np.uint8)        # 0 / 1

        sel = np.stack([s2[C.S2_BAND_IDX[b]] for b in BANDS])
        sel = np.clip(sel / S2_SCALE, 0, 1.0)
        sel = resize_bands(sel, C.CHIP_SIZE)

        lab_r = resize_mask(lab, C.CHIP_SIZE)
        jrc_r = resize_mask(jrc, C.CHIP_SIZE)
        y = np.where(lab_r < 0, 255, lab_r).astype(np.uint8)  # 255 = ignore

        valid = y != 255
        if valid.sum() < 0.2 * y.size:      # mostly no-data chip
            skipped += 1
            continue

        water_frac = float((y[valid] == 1).mean())
        perm_frac = float(jrc_r[valid].mean())
        flood_frac = max(0.0, water_frac - perm_frac)

        lat, lon = geo_centroid(f)
        event, chip_id = stem.split("_")[0], stem.split("_")[1]

        rgb8 = stretch_rgb(sel[:3])                    # (H,W,3) uint8
        Image.fromarray(rgb8).save(prev_dir / f"{stem}.png")
        RGB.append(rgb8.transpose(2, 0, 1))            # (3,H,W)

        X.append(sel.astype(np.float16))
        Y.append(y)
        J.append(jrc_r)
        rows.append(dict(
            chip=stem, event=event, chip_id=chip_id, lat=round(lat, 5), lon=round(lon, 5),
            water_frac=round(water_frac, 5), perm_water_frac=round(perm_frac, 5),
            flood_frac=round(flood_frac, 5),
            is_flooded=int(flood_frac > C.FLOOD_FRAC_THRESHOLD),
            valid_frac=round(float(valid.mean()), 4),
            preview=f"datasets/processed/images/preview/{stem}.png",
        ))
        if n % 50 == 0 or n == len(files):
            print(f"  [{n:4d}/{len(files)}] kept={len(rows)} skipped={skipped}")

    X = np.stack(X)
    Y = np.stack(Y)
    J = np.stack(J)
    np.save(C.PROC_IMG / "X.npy", X)
    np.save(C.PROC_IMG / "Y.npy", Y)
    np.save(C.PROC_IMG / "JRC.npy", J)
    RGB = np.stack(RGB)
    np.save(C.PROC_IMG / "RGB8.npy", RGB)
    meta = pd.DataFrame(rows)
    meta.to_csv(C.PROC / "chip_meta.csv", index=False)

    print(f"\nX {X.shape} {X.dtype}  ({X.nbytes / 1e6:.0f} MB)")
    print(f"Y {Y.shape}   JRC {J.shape}   RGB8 {RGB.shape} ({RGB.nbytes/1e6:.0f} MB)")
    print(f"Flooded chips: {meta.is_flooded.sum()} / {len(meta)} "
          f"({meta.is_flooded.mean() * 100:.1f}%)")
    print("\nPer event:")
    g = meta.groupby("event").agg(n=("chip", "size"), flooded=("is_flooded", "sum"),
                                  mean_water=("water_frac", "mean"),
                                  mean_flood=("flood_frac", "mean"))
    print(g.to_string())
    print(f"\nWrote {C.PROC / 'chip_meta.csv'} and arrays in {C.PROC_IMG}")


if __name__ == "__main__":
    main()
