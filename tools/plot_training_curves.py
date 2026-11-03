#!/usr/bin/env python3
"""Plot BasicSR training logs for all ZFC ablation runs.

Usage:
  python tools/plot_training_curves.py experiments/run_a/train.log experiments/run_b/train.log --out-dir results/training_curves
The parser extracts iteration, training losses and validation PSNR/SSIM where present.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ITER_RE = re.compile(r"\biter\s*[:=]\s*(\d+)")
PAIR_RE = re.compile(r"#\s*([A-Za-z0-9_]+)\s*:\s*([-+0-9.eE]+)")
LOSS_RE = re.compile(r"\b(l_[A-Za-z0-9_]+)\s*:\s*([-+0-9.eE]+)")


def parse_log(path):
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        it = ITER_RE.search(line)
        if not it:
            continue
        row = {"iter": int(it.group(1))}
        for key, value in PAIR_RE.findall(line):
            try:
                row[key] = float(value)
            except ValueError:
                pass
        for key, value in LOSS_RE.findall(line):
            try:
                row[key] = float(value)
            except ValueError:
                pass
        if len(row) > 1:
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = {Path(p).stem: parse_log(p) for p in args.logs}
    metrics = sorted({k for rows in runs.values() for row in rows for k in row if k != "iter"})
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        any_data = False
        for name, rows in runs.items():
            pts = [(r["iter"], r[metric]) for r in rows if metric in r]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], label=name)
                any_data = True
        if any_data:
            ax.set_xlabel("Iteration")
            ax.set_ylabel(metric)
            ax.set_title(f"Training / validation: {metric}")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f"{metric}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
