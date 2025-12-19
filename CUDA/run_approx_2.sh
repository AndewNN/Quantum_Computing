#!/bin/bash

# python PO_new_ApproxRatio.py -Q 2 -A 3 4 5 6 7 -E 35 -p 5 -B 24 -m Preserving -L 0.00005 -q 0.15
# python PO_new_ApproxRatio.py -Q 2 -A 10 -E 35 -p 5 -B 24 -m Preserving -L 0.00005 -q 0.15

# python PO_new_ApproxRatio.py -Q 2 -A 3 4 5 6 7 -E 35 -p 5 -m X -L 0.05 -q 0.15
# python PO_new_ApproxRatio.py -Q 2 -A 3 4 5 6 7 -E 35 -p 5 -m X -L 0.005 -q 0.15

python PO_new_ApproxRatio.py -Q 2 -A 3 4 5 6 7 -E 35 -p 5 -m X -L 0.05 -q 0.15 -b_X 250
