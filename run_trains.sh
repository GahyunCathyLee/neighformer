#!/usr/bin/env bash
set +e

python3 train.py --config configs/baseline.yaml

python3 train.py --config configs/imp_exp1.yaml

python3 train.py --config configs/imp_exp2.yaml

python3 train.py --config configs/imp_exp3.yaml