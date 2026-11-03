#!/usr/bin/env python3
"""Clone the paper's exact YAML and change only the ZFC ablation selector."""
import argparse
import copy
from pathlib import Path
import yaml


MODES = ("none", "se_matched", "film", "zfc")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "network_g" not in config:
        raise ValueError("Base YAML must parse to a mapping containing network_g.")
    config = copy.deepcopy(config)
    config["name"] = args.run_name
    config["network_g"]["dc_mode"] = args.mode
    config["network_g"]["enable_dc"] = args.mode != "none"

    # All other YAML fields are intentionally preserved.
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"Created {out} | mode={args.mode} | run={args.run_name}")


if __name__ == "__main__":
    main()
