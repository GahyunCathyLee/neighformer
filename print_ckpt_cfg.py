#!/usr/bin/env python3
"""Print the config stored inside a checkpoint file."""
import sys
import yaml
import torch


def main():
    if len(sys.argv) < 2:
        print("Usage: python print_ckpt_cfg.py <checkpoint.pt>")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # new format: cfg stored under "cfg" key
    if "cfg" in ckpt:
        cfg = ckpt["cfg"]
        print(yaml.dump(cfg, allow_unicode=True, default_flow_style=False), end="")
    else:
        # old format: config fields stored as top-level keys
        CONFIG_KEYS = {"cond", "feature_mode", "seed", "epoch", "val_rmse", "val_ade"}
        info = {k: ckpt[k] for k in CONFIG_KEYS if k in ckpt}
        if not info:
            print(f"[ERROR] No config keys found. Available keys: {list(ckpt.keys())}")
            sys.exit(1)
        print(yaml.dump(info, allow_unicode=True, default_flow_style=False), end="")


if __name__ == "__main__":
    main()
