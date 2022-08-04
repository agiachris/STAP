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

function train_policy {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --agent-config ${AGENT_CONFIG}"
    args="${args} --env-config ${ENV_CONFIG}"
    # args="${args} --encoder-checkpoint ${ENCODER_CHECKPOINT}"
    args="${args} --path ${OUTPUT_PATH}/${EXP_NAME}"
    args="${args} --seed 0"
    # args="${args} --overwrite"

    CMD="python scripts/train/train_policy.py ${args}"
    run_cmd
}

OUTPUT_PATH="models"
EXP_NAME="20220727/decoupled_state"
TRAINER_CONFIG="configs/pybox2d/trainers/agent.yaml"
AGENT_CONFIG="configs/pybox2d/agents/sac.yaml"

ENV_CONFIG="configs/pybox2d/envs/placeright.yaml"
train_policy

ENV_CONFIG="configs/pybox2d/envs/pushleft.yaml"
train_policy