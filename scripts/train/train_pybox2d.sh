#!/bin/bash
# Note: Conda environment must be manually activated
# 1) source <~/.bashrc or ~/.zshrc>
# 2) conda activate temporal_policies

output="false"
config="false"
while getopts p:c: flag
do
    case "${flag}" in
        p) output=${OPTARG};;
        c) config=${OPTARG};;
    esac
done

if [ $output == "false" ] || [ $config == "false" ]; then
    printf "Must pass -p path/to/save/dir and -c path/to/config/dir"
    exit
fi
output=$(realpath ${output})
config=$(realpath ${config})

i=0
inputs=()
outputs=()
for exp in "$@"; do
    if [ $i -gt 3 ]; then
        inputs+=($(realpath "${config}/${exp}.yaml"))
        outputs+=($(realpath "${output}/$(basename ${config})_${exp}"))
    fi
    i=$((i + 1))
done

len=${#inputs[@]}
for (( j=0; j < len; j++)); do
    eval "python scripts/train.py --path ${outputs[$j]} --config ${inputs[$j]}"
done
