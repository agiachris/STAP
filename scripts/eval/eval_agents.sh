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

function eval_policy {
    args=""
    args="${args} --checkpoint ${POLICY_CHECKPOINT}"
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path plots/${EXP_NAME}_debug"
        args="${args} --num-episodes 100"
        args="${args} --verbose 1"
    else
        args="${args} --path plots/${EXP_NAME}"
        args="${args} --verbose 0"
        args="${args} --num-episodes ${NUM_EPISODES}"
    fi
    if [[ -n "${DEBUG_RESULTS}" ]]; then
        args="${args} --debug-results ${DEBUG_RESULTS}"
    fi
    if [[ -n "${ENV_CONFIG}" ]]; then
        args="${args} --env-config ${ENV_CONFIG}"
    fi
    CMD="python scripts/eval/eval_agents.py ${args}"
    run_cmd
}

function run_policy {
    ENV_CONFIG="${ENV_CONFIG_PATH}/${PRIMITIVE}_eval.yaml"
    POLICY_CHECKPOINT="${POLICY_INPUT_PATH}/${PRIMITIVE}/${CHECKPOINT}.pt"
    EXP_NAME="${CHECKPOINT}"
    eval_policy
}

# Setup.
SBATCH_SLURM="scripts/train/train_juno.sh"
DEBUG=0

input_path="models"
output_path="plots"

# Pybullet experiments.
if [[ `hostname` == *stanford.edu ]] || [[ `hostname` == juno* ]]; then
    ENV_KWARGS="--gui 0"
fi

ENV_CONFIG_PATH="configs/pybullet/envs/official/primitives/light"
POLICY_INPUT_PATH="${input_path}/agents_rl"

# Evaluate skills.
NUM_EPISODES=100
CHECKPOINT="official_model"

PRIMITIVE="pick"
run_policy

PRIMITIVE="place"
run_policy

PRIMITIVE="pull"
run_policy

PRIMITIVE="push"
run_policy
