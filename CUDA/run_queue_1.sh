#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cudaq
export CUDA_VISIBLE_DEVICES=1

python PO_Mixer_Benchmark.py -Q 17 -A 3 -B 25 -st 25 -ed 50
python PO_Mixer_Benchmark.py -Q 17 -A 4 -B 25 -st 25 -ed 50
python PO_Mixer_Benchmark.py -Q 17 -A 5 -B 25 -st 25 -ed 50