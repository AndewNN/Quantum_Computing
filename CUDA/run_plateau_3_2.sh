#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cudaq
export CUDA_VISIBLE_DEVICES=3

export RUN_A=3
export RUN_Q=2
export RUN_L=0.002

for i in {500..4000..500} # N
do
    for j in {5..19..2} # Qubits
    do
        python PO_X_Plateau.py -Q $j -A $RUN_A -N $i -q $RUN_Q -L $RUN_L
    done
done