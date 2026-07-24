#!/bin/bash

for assets in 3 4 5 6 7 8
do
    for lambda in 0.0005 0.005 0.05 0.5 5
    do
        for layer in 5 7 9
        do
            python PO_new_ApproxRatio.py -Q 2 -A $assets -q 1.5 -m X -L $lambda -b_X 0 -E 10 -p $layer -norm Jh -d_b 0.3 -d_b 0.6 --LR_init
        done
    done
done
                
# for assets in 7
for assets in 3 4 5 6 7
do
    for layer in 5 7 9
    do
        python PO_new_ApproxRatio.py -Q 2 -A $assets -q 1.5 -m Preserving -L 1 -b_P 0 -E 10 -p $layer -norm Jh -d_b 0.3 -d_b 0.6 --LR_init
    done
done