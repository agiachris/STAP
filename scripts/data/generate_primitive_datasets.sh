#!/bin/bash

set -e

function run_cmd {
    echo ""
    echo "${CMD}"
    path_mod="./scripts/train:./scripts/eval:/.configs"
    if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == juno* ]] ; then
        sbatch scripts/data/generate_data_juno.sh "${path_mod}" "${CMD}"
    else
        export PYTHONPATH="${path_mod}:${PYTHONPATH}"
        pipenv run ${CMD}
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
NUM_PRETRAIN_STEPS="100000"
NUM_ENV_PROCESSES="2"

# Pybullet.
primitives=("pick" "place" "pull" "push")
train_seeds=("0" "1" "2" "3" "4" "5" "6" "7")
validation_seeds=("8" "9")
for PRIMITIVE in "${primitives[@]}"; do
    SPLIT="train"
    for SEED in "${train_seeds[@]}"; do
        generate_data
    done
    
    SPLIT="validation"
    for SEED in "${validation_seeds[@]}"; do
        generate_data
    done
done
