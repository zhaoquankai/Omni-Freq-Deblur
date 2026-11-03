#!/usr/bin/env python3
"""Create a temporary BasicSR YAML for one ZFC/SE/FiLM ablation run.

The source YAML is never modified. All variants retain the same data, losses,
optimizer and augmentation settings unless explicitly overridden here.
"""
import argparse
import copy
import os

import yaml


MODES = (
    "none",
    "zfc",
    "se",
    "se_matched",
    "cross_se_matched",
    "film",
    "film_bottleneck",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-iter", type=int, default=None)
    parser.add_argument("--num-gpu", type=int, default=1)
    parser.add_argument("--batch-size-per-gpu", type=int, default=None)
    parser.add_argument("--gt-size", type=int, default=None)
    parser.add_argument("--val-freq", type=int, default=None)
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.base_config, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)

    if not isinstance(opt, dict):
        raise ValueError(f"Invalid YAML config: {args.base_config}")
    if "network_g" not in opt or "train" not in opt:
        raise KeyError("Base config must contain 'network_g' and 'train' sections.")

    opt = copy.deepcopy(opt)
    base_name = opt.get("name", "omni_freq_deblur")
    opt["name"] = f"{base_name}_dc-{args.mode}_seed{args.seed}"
    opt["manual_seed"] = args.seed
    opt["num_gpu"] = args.num_gpu

    # This switch is the architecture entry point. The legacy enable_dc flag
    # remains synchronized so old configurations continue to behave correctly.
    opt["network_g"]["dc_mode"] = args.mode
    opt["network_g"]["enable_dc"] = args.mode != "none"

    # Every ablation is trained independently from scratch by default.
    opt.setdefault("path", {})
    opt["path"]["pretrain_network_g"] = None
    opt["path"]["resume_state"] = None

    if args.total_iter is not None:
        if args.total_iter <= 0:
            raise ValueError("--total-iter must be positive.")
        opt["train"]["total_iter"] = args.total_iter
        scheduler = opt["train"].get("scheduler")
        if isinstance(scheduler, dict) and "T_max" in scheduler:
            scheduler["T_max"] = args.total_iter

    train_dataset = opt.get("datasets", {}).get("train", {})
    if args.batch_size_per_gpu is not None:
        train_dataset["batch_size_per_gpu"] = args.batch_size_per_gpu
    if args.gt_size is not None:
        train_dataset["gt_size"] = args.gt_size

    if "val" in opt:
        if args.val_freq is not None:
            opt["val"]["val_freq"] = args.val_freq
        elif args.total_iter is not None and "val_freq" in opt["val"]:
            opt["val"]["val_freq"] = min(
                int(opt["val"]["val_freq"]),
                max(1, args.total_iter // 5)
            )

    if "logger" in opt and "save_checkpoint_freq" in opt["logger"]:
        if args.save_freq is not None:
            opt["logger"]["save_checkpoint_freq"] = args.save_freq
        elif args.total_iter is not None:
            opt["logger"]["save_checkpoint_freq"] = min(
                int(opt["logger"]["save_checkpoint_freq"]),
                max(1, args.total_iter // 5)
            )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.safe_dump(opt, f, sort_keys=False, allow_unicode=True)

    print(f"[Ablation Config] mode={args.mode}")
    print(f"[Ablation Config] seed={args.seed}")
    print(f"[Ablation Config] num_gpu={args.num_gpu}")
    print(f"[Ablation Config] total_iter={opt['train'].get('total_iter')}")
    print(f"[Ablation Config] output={args.output}")


if __name__ == "__main__":
    main()
