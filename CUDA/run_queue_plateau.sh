#! /bin/bash

# X mixer
# for j in {1..15..2} # p
# do
#     python PO_new_Plateau.py -Q 2 -A 2 3 4 5 6 -N 1000 -E 20 -p $j -Z 2 3
# done



# Preserving mixer
for i in {750..750..125} # N
do
    for j in {1..15..2} # p
    do
        python PO_new_Plateau.py -Q 2 -A 2 3 4 5 6 -N $i -E 20 -p $j -Z 2 3 -m Preserving -B 12
        # echo "Running N=$i, p=$j"
    done
done