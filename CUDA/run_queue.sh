#!/bin/bash
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cudaq
# export CUDA_VISIBLE_DEVICES=0


python PO_Mixer_Benchmark.py -Q 10 -A 3 -B 3 6 12
python PO_Mixer_Benchmark.py -Q 10 -A 4 -B 3 6 12
python PO_Mixer_Benchmark.py -Q 10 -A 5 -B 3 6 12

python PO_Mixer_Benchmark.py -Q 11 -A 3 -B 3 6 12
python PO_Mixer_Benchmark.py -Q 11 -A 4 -B 3 6 12
python PO_Mixer_Benchmark.py -Q 11 -A 5 -B 3 6 12

python PO_Mixer_Benchmark.py -Q 12 -A 3 -B 3 6 12
python PO_Mixer_Benchmark.py -Q 12 -A 4 -B 3 6 12
python PO_Mixer_Benchmark.py -Q 12 -A 5 -B 3 6 12

python PO_Mixer_Benchmark.py -Q 14 -A 3 -B 3 6 12
python PO_Mixer_Benchmark.py -Q 14 -A 4 -B 3 6 12
python PO_Mixer_Benchmark.py -Q 14 -A 5 -B 3 6 12