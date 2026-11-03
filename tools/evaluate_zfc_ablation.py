#!/usr/bin/env python3
"""Evaluate one ZFC ablation checkpoint and export metrics + diagnostic figures.

Required:
  --config options/train/GSblur.yml
  --checkpoint path/to/net_g_*.pth
  --lq-dir path/to/input
  --gt-dir path/to/target

Outputs are written under --out-dir:
  metrics_per_image.csv, metrics_summary.json, diagnostics/<image>_*.png

Optional dependencies:
  pip install scikit-image lpips
LPIPS is skipped if unavailable. NIQE/BRISQUE are intentionally not silently
substituted: they require an explicitly chosen implementation and calibration.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from basicsr.models.archs import define_network


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_rgb(path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def ssim(pred, target):
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return float("nan")
    p = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    t = target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return float(structural_similarity(p, t, channel_axis=2, data_range=1.0))


def spectral_metrics(pred, target):
    """Global FFT amplitude error and radial low/mid/high frequency errors."""
    p = pred.float()
    t = target.float()
    fp = torch.fft.fftshift(torch.fft.fft2(p, norm="ortho"), dim=(-2, -1))
    ft = torch.fft.fftshift(torch.fft.fft2(t, norm="ortho"), dim=(-2, -1))
    ap = torch.log1p(torch.abs(fp))
    at = torch.log1p(torch.abs(ft))
    h, w = p.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=p.device),
        torch.linspace(-1, 1, w, device=p.device),
        indexing="ij"
    )
    radius = torch.sqrt(xx * xx + yy * yy) / math.sqrt(2.0)
    bands = {
        "low": radius < (1.0 / 3.0),
        "mid": (radius >= (1.0 / 3.0)) & (radius < (2.0 / 3.0)),
        "high": radius >= (2.0 / 3.0)
    }
    error = torch.abs(ap - at)
    result = {"fft_log_amplitude_mae": float(error.mean().item())}
    for name, mask in bands.items():
        result[f"fft_{name}_mae"] = float(error[..., mask].mean().item())
        result[f"gt_{name}_energy"] = float(torch.abs(ft)[..., mask].mean().item())
        result[f"pred_{name}_energy"] = float(torch.abs(fp)[..., mask].mean().item())
    return result, ap[0].mean(0), at[0].mean(0)


def feature_statistics(feature):
    x = feature.detach().float()
    return {
        "feature_mean": float(x.mean().item()),
        "feature_std": float(x.std(unbiased=False).item()),
        "feature_l1": float(x.abs().mean().item()),
        "feature_l2": float(torch.sqrt(torch.mean(x * x)).item()),
        "feature_sparsity": float((x.abs() < 1e-6).float().mean().item())
    }


def save_heatmap(array, path, title, cmap="magma"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(array, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_diagnostics(lq, gt, pred, debug, image_stem, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    p = pred[0].detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    t = gt[0].detach().float().cpu().permute(1, 2, 0).numpy()
    b = lq[0].detach().float().cpu().permute(1, 2, 0).numpy()
    residual = np.abs(p - t).mean(axis=2)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, image, title in zip(axes, [b, p, t, residual],
                                ["Blurred input", "Restored", "Ground truth", "Absolute error"]):
        ax.imshow(image, cmap="magma" if image.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{image_stem}_restoration.png", dpi=180)
    plt.close(fig)

    pre = debug["l3_dec_pre_dc"][0].detach().float().cpu()
    post = debug["l3_dec_post_dc"][0].detach().float().cpu()
    delta = debug["modulation_delta"][0].detach().float().cpu()
    # Channel-averaged spatial response and RMS modulation magnitude.
    maps = {
        "l3_pre_dc": pre.abs().mean(0).numpy(),
        "l3_post_dc": post.abs().mean(0).numpy(),
        "zfc_delta": delta.abs().mean(0).numpy(),
        "zfc_delta_rms": torch.sqrt(torch.mean(delta * delta, dim=0)).numpy()
    }
    for name, array in maps.items():
        save_heatmap(array, out_dir / f"{image_stem}_{name}.png", name)

    gamma = debug["gamma"][0].detach().float().flatten().cpu().numpy()
    beta = debug["beta"][0].detach().float().flatten().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(gamma, bins=40)
    axes[0].set_title("Channel scale gamma")
    axes[1].hist(beta, bins=40)
    axes[1].set_title("Channel shift beta")
    fig.tight_layout()
    fig.savefig(out_dir / f"{image_stem}_modulation_hist.png", dpi=180)
    plt.close(fig)

    return {
        "gamma_mean": float(gamma.mean()),
        "gamma_std": float(gamma.std()),
        "gamma_min": float(gamma.min()),
        "gamma_max": float(gamma.max()),
        "beta_mean": float(beta.mean()),
        "beta_std": float(beta.std()),
        "beta_min": float(beta.min()),
        "beta_max": float(beta.max()),
        "modulation_delta_l1": float(delta.abs().mean().item()),
        "modulation_delta_l2": float(torch.sqrt(torch.mean(delta * delta)).item())
    }


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get("params", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lq-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--lpips", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    with open(args.config, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)
    net_opt = dict(opt["network_g"])
    mode = net_opt.get("dc_mode", "zfc")
    model = define_network(net_opt).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    lpips_model = None
    if args.lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex").to(device).eval()
        except Exception as exc:
            print(f"[WARN] LPIPS unavailable: {exc}")

    lq_dir, gt_dir = Path(args.lq_dir), Path(args.gt_dir)
    files = sorted(p for p in lq_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.max_images > 0:
        files = files[:args.max_images]
    out_dir = Path(args.out_dir)
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.inference_mode():
        for idx, lq_path in enumerate(files):
            gt_path = gt_dir / lq_path.name
            if not gt_path.exists():
                print(f"[WARN] Missing GT for {lq_path.name}; skip")
                continue
            lq, gt = load_rgb(lq_path).to(device), load_rgb(gt_path).to(device)
            _, _, h, w = lq.shape
            pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
            x = F.pad(lq, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else lq
            output, debug = model(x, return_debug=True)
            pred = output[..., :h, :w].clamp(0, 1)
            row = {"image": lq_path.name, "mode": mode,
                   "psnr": psnr(pred, gt), "ssim": ssim(pred, gt)}
            spec, pred_fft, gt_fft = spectral_metrics(pred, gt)
            row.update(spec)
            row.update({f"l3_pre_{k}": v for k, v in feature_statistics(debug["l3_dec_pre_dc"]).items()})
            row.update({f"l3_post_{k}": v for k, v in feature_statistics(debug["l3_dec_post_dc"]).items()})
            row.update(save_diagnostics(lq, gt, pred, debug, lq_path.stem, diag_dir)
                       if idx % max(1, args.save_every) == 0 else {})
            if lpips_model is not None:
                lp = lpips_model(pred * 2 - 1, gt * 2 - 1).mean().item()
                row["lpips_alex"] = float(lp)
            rows.append(row)
            print(f"[{idx + 1}/{len(files)}] {lq_path.name} | "
                  f"PSNR={row['psnr']:.4f} | SSIM={row['ssim']:.5f} | "
                  f"FFT-MAE={row['fft_log_amplitude_mae']:.6f}")

    if not rows:
        raise RuntimeError("No paired images were evaluated.")
    keys = sorted({key for row in rows for key in row})
    with open(out_dir / "metrics_per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    numeric = [k for k in keys if k not in ("image", "mode")]
    summary = {
        "mode": mode,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_images": len(rows),
        "metrics": {
            key: {
                "mean": float(np.nanmean([r.get(key, np.nan) for r in rows])),
                "std": float(np.nanstd([r.get(key, np.nan) for r in rows]))
            } for key in numeric
        }
    }
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
