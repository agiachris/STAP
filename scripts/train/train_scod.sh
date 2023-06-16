#!/bin/bash

set -e

function run_cmd {
    echo ""
    echo "${CMD}"
    if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == juno* ]]; then
        sbatch "${SBATCH_SLURM}" "${CMD}"
    else
        ${CMD}
    fi
}

function train_scod {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --scod-config ${SCOD_CONFIG}"
    args="${args} --model-checkpoint ${POLICY_CHECKPOINT}"
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"

    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path ${SCOD_OUTPUT_PATH}_debug"
        args="${args} --overwrite"
    else
        args="${args} --path ${SCOD_OUTPUT_PATH}"
    fi

    CMD="python scripts/train/train_scod.py ${args}"
    run_cmd
}

function run_scod {
    POLICY_CHECKPOINT="${POLICY_CHECKPOINT_PATH}/${PRIMITIVE}/${SELECTED_POLICY_CHECKPOINT}.pt"
    SCOD_OUTPUT_PATH="${SCOD_OUTPUT_DIR}/${SELECTED_POLICY_CHECKPOINT}/${PRIMITIVE}"
    train_scod
}

# Setup.
SBATCH_SLURM="scripts/train/train_juno.sh"
DEBUG=0

input_path="models"
output_path="models"

# Pybullet experiments.
if [[ `hostname` == *stanford.edu ]] || [[ `hostname` == juno* ]]; then
    ENV_KWARGS="--gui 0"
fi

# Train SCOD.
exp_name="scod"
TRAINER_CONFIG="configs/pybullet/trainers/uncertainty/scod.yaml"
SCOD_CONFIG="configs/pybullet/uncertainty/scod.yaml"

SCOD_OUTPUT_DIR="${output_path}/${exp_name}"
POLICY_CHECKPOINT_PATH="${input_path}/primitives_light_mse"
SELECTED_POLICY_CHECKPOINT="official_model"

PRIMITIVE="pick"
run_scod

PRIMITIVE="place"
run_scod

PRIMITIVE="pull"
run_scod

PRIMITIVE="push"
run_scod