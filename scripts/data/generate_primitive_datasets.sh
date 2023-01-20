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
    args="--config.exp-name ${EXP_NAME}"
    args="${args} --config.split ${SPLIT}"
    args="${args} --config.primitive ${PRIMITIVE}"
    args="${args} --config.symbolic-action-type ${SYMBOLIC_ACTION_TYPE}"
    args="${args} --config.seed ${SEED}"
    args="${args} --config.num-pretrain-steps ${NUM_PRETRAIN_STEPS}"
    
    CMD="python scripts/data/generate_primitive_dataset.py ${args}"    
    run_cmd
}

function run_data_generation {
    SPLIT="train"
    for SEED in "${TRAIN_SEEDS[@]}"; do
        generate_data
    done
    
    SPLIT="validation"
    for SEED in "${VALIDATION_SEEDS[@]}"; do
        generate_data
    done
}

# Experiments.
EXP_NAME="20230119/datasets"
SYMBOLIC_ACTION_TYPE="valid"
NUM_PRETRAIN_STEPS="100000"

# Pybullet.
TRAIN_SEEDS=("0" "1" "2" "3" "4" "5" "6" "7")
VALIDATION_SEEDS=("8" "9")

PRIMITIVE="pick"
run_data_generation

PRIMITIVE="place"
run_data_generation

PRIMITIVE="pull"
run_data_generation

PRIMITIVE="push"
run_data_generation
