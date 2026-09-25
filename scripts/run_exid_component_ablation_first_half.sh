#!/usr/bin/env bash
set -euo pipefail

SHARD_INDEX=0 SHARD_COUNT=2 "$(dirname "$0")/run_exid_component_ablation.sh"
