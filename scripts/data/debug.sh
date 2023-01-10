#!/bin/bash

l1=("1" "2" "3" "4")
ls=("0" "0" "0" "0")
le=("5" "6" "7" "8")

for i in "${!l1[@]}"; do
    for j in $( seq ${ls[$i]} $((${le[$i]} - 1)); do
        echo "${j}"
    done
done
