#!/bin/bash

set -e

GCP_LOGIN="juno-login-lclbjqwy-001"

function run_cmd {
    echo ""
    echo "${CMD}"
    if [[ `hostname` == "sc.stanford.edu" ]]; then
        sbatch scripts/eval/eval_planners_juno.sh "${CMD}"
    elif [[ `hostname` == "${GCP_LOGIN}" ]]; then
        sbatch scripts/eval/eval_gcp.sh "${CMD}"
    else
        ${CMD}
    fi
}

function eval_policies {
    args=""
    args="${args} --checkpoint ${POLICY_CHECKPOINT}"
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path plots/${EXP_NAME}_debug"
        args="${args} --num-episodes 1"
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
    CMD="python scripts/eval/eval_policies.py ${args}"
    run_cmd
}

# Setup.

DEBUG=0
NUM_EPISODES=20

# Evaluate policies.

policy_envs=(
    "pick"
    "place"
    "pull"
    "push"
)
experiments=(
    "20220908/official"
)
ckpts=(
    # "best_model"
    # "final_model"
    # "ckpt_model_50000"
    # "ckpt_model_100000"
    # "ckpt_model_150000"
    # "ckpt_model_200000"
)
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]]; then
    ENV_KWARGS="${ENV_KWARGS} --gui 0"
fi

for EXP_NAME in "${experiments[@]}"; do
    for ckpt in "${ckpts[@]}"; do
        for policy_env in "${policy_envs[@]}"; do
            POLICY_CHECKPOINT="models/${EXP_NAME}/${policy_env}/${ckpt}.pt"
            eval_policies
        done
    done
done
