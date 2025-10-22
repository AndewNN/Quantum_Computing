#! /bin/bash

python PO_Mixer_Benchmark.py -Q 4 -A 3 -B 3 -m X
python PO_Mixer_Benchmark.py -Q 4 -A 3 -B 3 -st 0 -ed 25 -m Preserving
python PO_Mixer_Benchmark.py -Q 4 -A 3 -B 3 -st 25 -ed 50 -m Preserving
python PO_Mixer_Benchmark.py -Q 4 -A 3 -B 3 -st 50 -ed 75 -m Preserving
python PO_Mixer_Benchmark.py -Q 4 -A 3 -B 3 -st 75 -ed 100 -m Preserving