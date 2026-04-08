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

    print(f"[INFO] keys in checkpoint: {list(ckpt.keys())}")

    cfg = ckpt.get("cfg")
    if cfg is None:
        print("[ERROR] 'cfg' key not found in checkpoint.")
        sys.exit(1)

    print(yaml.dump(cfg, allow_unicode=True, default_flow_style=False), end="")


if __name__ == "__main__":
    main()
