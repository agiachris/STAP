#!/bin/bash

set -e

function run_cmd {
    echo ""
    echo "${CMD}"
    if [[ `hostname` == "sc.stanford.edu" ]]; then
        sbatch scripts/train/train_juno.sh "${CMD}"
    else
        ${CMD}
    fi
}

function train_scod {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --scod-config ${SCOD_CONFIG}"
    args="${args} --model-checkpoint ${OUTPUT_PATH}/${MODEL_CHECKPOINT}"
    args="${args} --model-network ${MODEL_NETWORK}"
    args="${args} --path ${OUTPUT_PATH}/${EXP_NAME}"
    args="${args} --seed 0"

    CMD="python scripts/train/train_scod.py ${args}"
    run_cmd
}


OUTPUT_PATH="${OUTPUTS}/temporal_policies/models"
TRAINER_CONFIG="configs/pybox2d/trainers/scod.yaml"
SCOD_CONFIG="configs/pybox2d/scod/scod.yaml"
MODEL_NETWORK="critic"

EXP_NAME="20220718/decoupled_state/placeright"
MODEL_CHECKPOINT="${EXP_NAME}/final_model.pt"
train_scod

EXP_NAME="20220718/decoupled_state/pushleft"
MODEL_CHECKPOINT="${EXP_NAME}/final_model.pt"
train_scod
