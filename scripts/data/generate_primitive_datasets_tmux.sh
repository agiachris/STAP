#!/bin/bash

set -e

function run_cmd {
    echo ""
    path_mod="./scripts/train:./scripts/eval:/.configs"
    tmux_name="${SPLIT}_${PRIMITIVE}_${SEED}_${CPU}"

    tmux new-session -d -s "${tmux_name}"
    tmux send-keys -t "${tmux_name}" "export PYTHONPATH=${path_mod}:${PYTHONPATH}" Enter
    tmux send-keys -t "${tmux_name}" "taskset -c ${CPU} pipenv run ${PYTHON_CMD}" Enter
}

function generate_data {
    args="--config.exp-name ${EXP_NAME}"
    args="${args} --config.split ${SPLIT}"
    args="${args} --config.primitive ${PRIMITIVE}"
    args="${args} --config.symbolic-action-type ${SYMBOLIC_ACTION_TYPE}"
    args="${args} --config.seed ${SEED}"
    
    PYTHON_CMD="python scripts/data/generate_primitive_dataset.py ${args}"
    run_cmd
}

function run_data_generation {
    for idx in "${!SEEDS[@]}"; do
        SEED="${SEEDS[${idx}]}"
        CPU="${CPUS[${idx}]}"
        generate_data
    done
}

# Experiments.
EXP_NAME="20230124/datasets"
SYMBOLIC_ACTION_TYPE="valid"

# Pybullet.
SPLIT="train"
SEEDS=($(seq 0 7))
CPUS=($(seq 4 11))
PRIMITIVE="push"

run_data_generation