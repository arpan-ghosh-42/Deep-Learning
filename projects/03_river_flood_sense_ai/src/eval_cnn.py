"""
Evaluate a saved U-Net checkpoint and refresh its metrics + per-chip predictions.

train_cnn.py writes its checkpoint every time validation improves, but only writes
metrics and the pred_water_frac_* column at the very end. If a training run is stopped
early, this script brings those artifacts back in sync with the best checkpoint that was
actually saved - using the identical split, so the numbers stay comparable.

    python src/eval_cnn.py --channels rgb
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from train_cnn import UNet, evaluate, load_data, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", choices=["ms", "rgb"], required=True)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    ck_path = C.MODELS / f"unet_water_{args.channels}.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    size = ck.get("size", 128)

    X, Y, meta = load_data(size, args.channels)
    tr, va, te = split(meta)

    model = UNet(cin=ck["in_ch"], base=ck["base"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    viou, vf1 = evaluate(model, X, Y, va, args.batch)
    tiou, tf1 = evaluate(model, X, Y, te, args.batch)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"{ck_path.name}  base={ck['base']} in_ch={ck['in_ch']} size={size} "
          f"{nparam/1e6:.2f}M params")
    print(f"  val  IoU {viou:.4f}  F1 {vf1:.4f}")
    print(f"  TEST IoU {tiou:.4f}  F1 {tf1:.4f}")

    fracs = np.zeros(len(meta))
    with torch.no_grad():
        for s in range(0, len(meta), args.batch):
            b = np.arange(s, min(s + args.batch, len(meta)))
            pr = (torch.sigmoid(model(X[b])) > 0.5).float()
            valid = (Y[b] != 255).float()
            fracs[b] = ((pr * valid).sum((1, 2)) /
                        valid.sum((1, 2)).clamp(min=1)).numpy()

    meta[f"pred_water_frac_{args.channels}"] = fracs.round(5)
    meta["split"] = "train"
    meta.loc[va, "split"] = "val"
    meta.loc[te, "split"] = "test"
    meta.to_csv(C.PROC / "chip_meta.csv", index=False)

    p = C.REPORTS / f"cnn_metrics_{args.channels}.json"
    old = json.load(open(p)) if p.exists() else {}
    old.update(test_iou=tiou, test_f1=tf1, best_val_iou=viou, params=nparam,
               size=size, channels=args.channels, base=ck["base"],
               note="metrics recomputed from the saved best checkpoint by eval_cnn.py")
    json.dump(old, open(p, "w"), indent=2)

    corr = meta[f"pred_water_frac_{args.channels}"].corr(meta.water_frac)
    t = meta[meta.split == "test"]
    print(f"  water-fraction corr: all r={corr:.3f}  "
          f"test r={t[f'pred_water_frac_{args.channels}'].corr(t.water_frac):.3f}")
    print(f"Updated {p.name} and chip_meta.csv")


if __name__ == "__main__":
    main()
