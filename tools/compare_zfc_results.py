#!/usr/bin/env python3
"""Combine per-image evaluation CSVs and compute paired method deltas vs ZFC."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["image"]: row for row in csv.DictReader(f)}


def bootstrap_ci(values, n_boot=10000, seed=2026):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zfc", required=True)
    parser.add_argument("--comparison", nargs="+", required=True,
                        help="NAME=metrics_per_image.csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    zfc = read_csv(args.zfc)
    comparisons = {}
    for item in args.comparison:
        name, path = item.split("=", 1)
        comparisons[name] = read_csv(path)

    metrics = ("psnr", "ssim", "lpips_alex", "fft_log_amplitude_mae",
               "fft_low_mae", "fft_mid_mae", "fft_high_mae", "gradient_mae")
    report = {"reference": "zfc", "num_zfc_images": len(zfc), "comparisons": {}}
    for name, rows in comparisons.items():
        common = sorted(set(zfc) & set(rows))
        result = {"num_paired_images": len(common), "metrics": {}}
        for metric in metrics:
            pairs = []
            for image in common:
                if metric in zfc[image] and metric in rows[image]:
                    try:
                        a, b = float(rows[image][metric]), float(zfc[image][metric])
                    except (ValueError, TypeError):
                        continue
                    if np.isfinite(a) and np.isfinite(b):
                        # Positive means comparison method has a higher value.
                        pairs.append(a - b)
            if pairs:
                result["metrics"][metric] = {
                    "comparison_minus_zfc_mean": float(np.mean(pairs)),
                    "paired_bootstrap_95ci": bootstrap_ci(pairs),
                    "n": len(pairs)
                }
        report["comparisons"][name] = result

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
