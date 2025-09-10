#!/bin/bash

python PO_Mixer_Benchmark.py -Q 4 -A 3 -B 3 -st 0 -ed 25
python PO_Mixer_Benchmark.py -Q 4 -A 3 -B 3 -st 25 -ed 50
python PO_Mixer_Benchmark.py -Q 4 -A 3 -B 3 -st 50 -ed 75
python PO_Mixer_Benchmark.py -Q 4 -A 3 -B 3 -st 75 -ed 100