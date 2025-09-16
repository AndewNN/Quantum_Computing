#!/bin/bash

# Usage: ./run_merge.sh Q A B
# Example: ./run_merge.sh 4 3 3

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cudaq

Q=$1
A=$2
B=$3

# Copy X from Xbase
python PO_Mixer_Benchmark.py -Q $Q -A $A -B $B -st 0  -ed 25  -m Preserving
python PO_Mixer_Benchmark.py -Q $Q -A $A -B $B -st 25 -ed 50  -m Preserving
python PO_Mixer_Benchmark.py -Q $Q -A $A -B $B -st 50 -ed 75  -m Preserving
python PO_Mixer_Benchmark.py -Q $Q -A $A -B $B -st 75 -ed 100 -m Preserving

# Merge from splits and Generate report
python PO_Mixer_Benchmark_Merger.py -Q $Q -A $A -B $B

