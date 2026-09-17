#!/usr/bin/env bash
set +e

#python3 train.py --config configs/highD0-1.yaml

#python3 train.py --config configs/highD0-2.yaml

#python3 train.py --config configs/highD0-3.yaml

#python3 train.py --config configs/highD0-4.yaml

#python3 train.py --config configs/highD0-5.yaml

python evaluate.py --ckpt ckpts/highD0-1/best.pt --scenario

python evaluate.py --ckpt ckpts/highD0-2/best.pt --scenario

python evaluate.py --ckpt ckpts/highD0-3/best.pt --scenario

python evaluate.py --ckpt ckpts/highD0-4/best.pt --scenario

python evaluate.py --ckpt ckpts/highD0-5/best.pt --scenario