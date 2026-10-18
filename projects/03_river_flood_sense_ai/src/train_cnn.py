"""
Train the satellite branch: a U-Net that segments open water in a Sentinel-2 chip.

Two input modes, because they serve two different users:

  --channels ms   7 channels: red, green, blue, NIR, SWIR1, NDWI, MNDWI
                  full Sentinel-2 - what you have if you downloaded the real product
  --channels rgb  3 channels: contrast-stretched red, green, blue
                  what you have if someone hands you an ordinary satellite PNG/JPEG.
                  Trained on RGB8.npy - the exact byte content of the preview images -
                  so the network sees at training time what the app feeds it at
                  inference time. Training the RGB model on raw reflectance instead
                  would silently break on every real upload.

Output : per-pixel water probability

NDWI  = (green - NIR)   / (green + NIR)     classic open-water index
MNDWI = (green - SWIR1) / (green + SWIR1)   better at separating water from built-up

Handing the network these two physical indices alongside the raw bands is what lets a
~450-image dataset train a usable water segmenter: the indices already encode most of the
water signal, and the U-Net learns shape, context and the cases the indices get wrong
(cloud shadow, turbid flood water, wet soil).

The model is fully convolutional, so it trains at 128x128 for speed and runs at any size
at inference time.

Usage:
    python src/train_cnn.py --epochs 40 --size 128
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

IGNORE = 255


# ---------------------------------------------------------------------------- model
def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, cin=7, base=24):
        super().__init__()
        b = base
        self.e1, self.e2, self.e3, self.e4 = block(cin, b), block(b, b * 2), block(b * 2, b * 4), block(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2)
        self.d3 = block(b * 8, b * 4)
        self.u2 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2)
        self.d2 = block(b * 4, b * 2)
        self.u1 = nn.ConvTranspose2d(b * 2, b, 2, 2)
        self.d1 = block(b * 2, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1).squeeze(1)


# ------------------------------------------------------------------------------ data
def spectral_indices(x):
    """x (N,5,H,W): red, green, blue, nir, swir1 -> append NDWI and MNDWI."""
    g, nir, swir = x[:, 1], x[:, 3], x[:, 4]
    eps = 1e-6
    ndwi = (g - nir) / (g + nir + eps)
    mndwi = (g - swir) / (g + swir + eps)
    return torch.cat([x, ndwi.unsqueeze(1), mndwi.unsqueeze(1)], 1)


def load_data(size, channels="ms"):
    if channels == "rgb":
        X = np.load(C.PROC_IMG / "RGB8.npy").astype(np.float32) / 255.0
    else:
        X = np.load(C.PROC_IMG / "X.npy").astype(np.float32)
    Y = np.load(C.PROC_IMG / "Y.npy")
    meta = pd.read_csv(C.PROC / "chip_meta.csv")
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y.astype(np.int64))
    if size != X.shape[-1]:
        X = F.interpolate(X, size=(size, size), mode="bilinear", align_corners=False)
        Y = F.interpolate(Y.unsqueeze(1).float(), size=(size, size), mode="nearest").squeeze(1).long()
    X = spectral_indices(X) if channels == "ms" else X
    return X, Y, meta


def split(meta, seed=42):
    """Stratified by flood event AND flooded/dry, so every split sees both."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for _, grp in meta.groupby(["event", "is_flooded"]):
        idx = np.array(grp.index.to_numpy(), copy=True)
        rng.shuffle(idx)
        n = len(idx)
        n_te = max(1, int(round(0.15 * n))) if n >= 4 else 0
        n_va = max(1, int(round(0.15 * n))) if n >= 4 else 0
        te += list(idx[:n_te])
        va += list(idx[n_te:n_te + n_va])
        tr += list(idx[n_te + n_va:])
    return np.array(tr), np.array(va), np.array(te)


def augment(x, y):
    if torch.rand(1).item() < 0.5:
        x, y = torch.flip(x, [-1]), torch.flip(y, [-1])
    if torch.rand(1).item() < 0.5:
        x, y = torch.flip(x, [-2]), torch.flip(y, [-2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        x, y = torch.rot90(x, k, [-2, -1]), torch.rot90(y, k, [-2, -1])
    return x, y


# ----------------------------------------------------------------------------- loss
def masked_loss(logits, target, pos_weight):
    valid = target != IGNORE
    if valid.sum() == 0:
        return logits.sum() * 0
    t = target.clone()
    t[~valid] = 0
    bce = F.binary_cross_entropy_with_logits(
        logits, t.float(), reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device))
    bce = (bce * valid).sum() / valid.sum()

    p = torch.sigmoid(logits) * valid
    tt = t.float() * valid
    dice = 1 - (2 * (p * tt).sum() + 1) / (p.sum() + tt.sum() + 1)
    return bce + dice


@torch.no_grad()
def evaluate(model, X, Y, idx, bs=8):
    model.eval()
    inter = union = tp = fp = fn = 0
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        logit = model(X[b])
        pred = (torch.sigmoid(logit) > 0.5)
        t = Y[b]
        valid = t != IGNORE
        p, g = pred & valid, (t == 1) & valid
        inter += (p & g).sum().item()
        union += (p | g).sum().item()
        tp += (p & g).sum().item()
        fp += (p & ~g).sum().item()
        fn += (~p & g).sum().item()
    iou = inter / max(union, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return iou, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--channels", choices=["ms", "rgb"], default="ms",
                    help="ms = 5 bands + NDWI + MNDWI; rgb = red/green/blue only")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    X, Y, meta = load_data(args.size, args.channels)
    tr, va, te = split(meta)
    print(f"chips: train={len(tr)} val={len(va)} test={len(te)}   input {tuple(X.shape[1:])}")

    water = (Y[tr] == 1).sum().item()
    land = (Y[tr] == 0).sum().item()
    pos_weight = min(5.0, max(1.0, land / max(water, 1)))
    print(f"water pixels {water / (water + land) * 100:.1f}%  -> pos_weight={pos_weight:.2f}\n")

    ckpt_path = C.MODELS / f"unet_water_{args.channels}.pt"
    model = UNet(cin=X.shape[1], base=args.base)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(tr) // args.batch))
    print(f"U-Net[{args.channels}] base={args.base}  {nparam/1e6:.2f}M params  epochs={args.epochs}\n")

    best, hist = -1, []
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(tr)
        tot, nb, t0 = 0.0, 0, time.time()
        for s in range(0, len(perm) - args.batch + 1, args.batch):
            b = perm[s:s + args.batch]
            xb, yb = augment(X[b], Y[b])
            opt.zero_grad()
            loss = masked_loss(model(xb), yb, pos_weight)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        viou, vf1 = evaluate(model, X, Y, va, args.batch)
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1), val_iou=viou, val_f1=vf1))
        flag = ""
        if viou > best:
            best = viou
            torch.save({"state_dict": model.state_dict(), "base": args.base,
                        "in_ch": X.shape[1], "size": args.size,
                        "channels": args.channels}, ckpt_path)
            flag = "  <- best"
        print(f"ep {ep:3d}/{args.epochs}  loss {tot/max(nb,1):.4f}  "
              f"val IoU {viou:.4f}  val F1 {vf1:.4f}  ({time.time()-t0:.0f}s){flag}")

    ck = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    tiou, tf1 = evaluate(model, X, Y, te, args.batch)
    print(f"\nTEST  water IoU {tiou:.4f}   F1 {tf1:.4f}")

    # Per-chip predicted water fraction: the number the fusion model consumes.
    model.eval()
    fracs = np.zeros(len(meta))
    with torch.no_grad():
        for s in range(0, len(meta), args.batch):
            b = np.arange(s, min(s + args.batch, len(meta)))
            pr = (torch.sigmoid(model(X[b])) > 0.5).float()
            valid = (Y[b] != IGNORE).float()
            fracs[b] = ((pr * valid).sum((1, 2)) / valid.sum((1, 2)).clamp(min=1)).numpy()
    meta[f"pred_water_frac_{args.channels}"] = fracs.round(5)
    meta["split"] = "train"
    meta.loc[va, "split"] = "val"
    meta.loc[te, "split"] = "test"
    meta.to_csv(C.PROC / "chip_meta.csv", index=False)

    json.dump(dict(test_iou=tiou, test_f1=tf1, best_val_iou=best, params=nparam,
                   epochs=args.epochs, size=args.size, history=hist,
                   channels=args.channels),
              open(C.REPORTS / f"cnn_metrics_{args.channels}.json", "w"), indent=2)
    print(f"Saved {ckpt_path} and cnn_metrics_{args.channels}.json")


if __name__ == "__main__":
    main()
