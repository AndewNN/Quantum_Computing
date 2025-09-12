#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cudaq

# Copy X from Xbase
python PO_Mixer_Benchmark.py -Q 4 -A 3 -B 3 -st 0 -ed 25 -m Preserving
python PO_Mixer_Benchmark.py -Q 4 -A 3 -B 3 -st 25 -ed 50 -m Preserving
python PO_Mixer_Benchmark.py -Q 4 -A 3 -B 3 -st 50 -ed 75 -m Preserving
python PO_Mixer_Benchmark.py -Q 4 -A 3 -B 3 -st 75 -ed 100 -m Preserving

# Merge from splits
python PO_Mixer_Benchmark_Merger.py -Q 4 -A 3 -B 3
# Generate report
python PO_Mixer_Benchmark.py -Q 4 -A 3 -B 3

