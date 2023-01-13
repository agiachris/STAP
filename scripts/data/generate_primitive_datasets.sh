#!/bin/bash

set -e

function run_cmd {
    echo ""
    echo "${CMD}"
    pythonpath="./scripts/train:./scripts/eval:./configs"
    if [[ `hostname` == "sc.stanford.edu" ]]; then
        sbatch scripts/data/generate_data_juno.sh "${pythonpath}" "${CMD}"
    else
        export PYTHONPATH="${pythonpath}:${PYTHONPATH}"
        eval ${CMD}
    fi
}

function generate_data {
    args=""
    # Data specific.
    args="${args} --config.split ${SPLIT}"
    args="${args} --config.primitive ${PRIMITIVE}"
    args="${args} --config.symbolic-action-type ${SYMBOLIC_ACTION_TYPE}"
    args="${args} --config.seed ${SEED}"
    # Compute specific.
    args="${args} --config.num-pretrain-steps ${NUM_PRETRAIN_STEPS}"
    args="${args} --config.num-env-processes ${NUM_ENV_PROCESSES}"
    
    CMD="python scripts/data/generate_primitive_dataset.py ${args}"
    run_cmd
}

# Experiments.
EXP_NAME="20230113/datasets"

SYMBOLIC_ACTION_TYPE="valid"
NUM_PRETRAIN_STEPS="200000"
NUM_ENV_PROCESSES="4"

# Pybullet.
primitives=("pick" "place" "pull" "push")
seeds=("0" "1" "2" "3")
SPLIT="train"
for PRIMITIVE in "${primitives[@]}"; do
    for SEED in "${seeds[@]}"; do
        generate_data
    done
done

SEED="4"
SPLIT="validation"
for PRIMITIVE in "${primitives[@]}"; do
    generate_data
done
