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
    args="${args} --path models/${EXP_NAME}"
    args="${args} --seed 0"
    # args="${args} --overwrite"

    CMD="python scripts/train/train_policy.py ${args}"
    run_cmd
}

EXP_NAME="20220427/decoupled"

TRAINER_CONFIG="configs/pybox2d/trainers/agent.yaml"
AGENT_CONFIG="configs/pybox2d/agents/sac_img.yaml"
ENV_CONFIG="configs/pybox2d/envs/placeright_img.yaml"
train_policy

TRAINER_CONFIG="configs/pybox2d/trainers/agent.yaml"
AGENT_CONFIG="configs/pybox2d/agents/sac_img.yaml"
ENV_CONFIG="configs/pybox2d/envs/pushleft_img.yaml"
train_policy
