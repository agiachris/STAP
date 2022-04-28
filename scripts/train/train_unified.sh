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

function train_unified {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --dynamics-config ${DYNAMICS_CONFIG}"
    args="${args} --agent-configs ${AGENT_CONFIGS}"
    args="${args} --env-configs ${ENV_CONFIGS}"
    if [ ! -z "${POLICY_CHECKPOINTS}" ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS}"
    fi
    args="${args} --path models/${EXP_NAME}"
    args="${args} --seed 0"
    # args="${args} --overwrite"

    CMD="python scripts/train/train_unified.py ${args}"
    run_cmd
}

EXP_NAME="20220427/unified"
TRAINER_CONFIG="configs/pybox2d/trainers/unified.yaml"
DYNAMICS_CONFIG="configs/pybox2d/dynamics/shared.yaml"
AGENT_CONFIGS="configs/pybox2d/agents/sac_img.yaml configs/pybox2d/agents/sac_img.yaml"
ENV_CONFIGS="configs/pybox2d/envs/placeright_img.yaml configs/pybox2d/envs/pushleft_img.yaml"
train_unified
