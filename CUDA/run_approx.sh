#!/bin/bash

python PO_new_ApproxRatio.py -Q 2 -A 3 4 5 6 7 -E 50 -p 5 -m X

python PO_new_ApproxRatio.py -Q 2 -A 3 4 5 6 -E 25 -p 5 -B 12 -m Preserving
python PO_new_ApproxRatio.py -Q 2 -A 3 4 5 6 -E 25 -p 5 -B 24 -m Preserving
python PO_new_ApproxRatio.py -Q 2 -A 3 4 5 6 -E 50 -p 5 -B 12 -m Preserving
python PO_new_ApproxRatio.py -Q 2 -A 3 4 5 6 -E 50 -p 5 -B 24 -m Preserving

python PO_new_ApproxRatio.py -Q 2 -A 7 -E 25 -p 5 -B 12 -m Preserving
python PO_new_ApproxRatio.py -Q 2 -A 7 -E 25 -p 5 -B 24 -m Preserving
python PO_new_ApproxRatio.py -Q 2 -A 7 -E 50 -p 5 -B 12 -m Preserving
python PO_new_ApproxRatio.py -Q 2 -A 7 -E 50 -p 5 -B 24 -m Preserving

python PO_new_ApproxRatio.py -Q 2 -A 8 -E 25 -p 5 -B 12 -m Preserving
python PO_new_ApproxRatio.py -Q 2 -A 8 -E 25 -p 5 -B 24 -m Preserving
python PO_new_ApproxRatio.py -Q 2 -A 8 -E 50 -p 5 -B 12 -m Preserving
python PO_new_ApproxRatio.py -Q 2 -A 8 -E 50 -p 5 -B 24 -m Preserving