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
    args="${args} --config.trainer-config ${TRAINER_CONFIG}"
    args="${args} --config.split ${SPLIT}"
    args="${args} --config.primitive ${PRIMITIVE}"
    args="${args} --config.symbolic-action-type ${SYMBOLIC_ACTION_TYPE}"
    args="${args} --config.seed ${SEED}"
    
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
EXP_NAME="20230125/datasets"

# Pybullet.

# In-distribution (symbolically valid) data generation.
# TRAINER_CONFIG="configs/pybullet/trainers/primitive_valid_dataset.yaml"
# SYMBOLIC_ACTION_TYPE="valid"
# TRAIN_SEEDS=($(seq 0 15))
# VALIDATION_SEEDS=($(seq 16 19))

# PRIMITIVE="pick"
# run_data_generation

# PRIMITIVE="place"
# run_data_generation

# PRIMITIVE="pull"
# run_data_generation

# PRIMITIVE="push"
# run_data_generation

# Out-of-distribution (symbolically invalid) data generation.
# TRAINER_CONFIG="configs/pybullet/trainers/primitive_invalid_dataset.yaml"
# SYMBOLIC_ACTION_TYPE="invalid"
# TRAIN_SEEDS=("0")
# VALIDATION_SEEDS=()

# PRIMITIVE="pick"
# run_data_generation

# PRIMITIVE="place"
# run_data_generation

# PRIMITIVE="pull"
# run_data_generation

# PRIMITIVE="push"
# run_data_generation