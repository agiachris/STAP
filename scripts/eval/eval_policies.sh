#!/bin/bash

set -e

function run_cmd {
    echo ""
    echo "${CMD}"
    if [[ `hostname` == "sc.stanford.edu" ]]; then
        sbatch scripts/eval/eval_planners_juno.sh "${CMD}"
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

# Evaluate policies.

policy_envs=(
    "pick"
    "place"
    "pull"
    "push"
)
experiments=(
    "20220905/official"
)
ckpts=(
    # "final_model"
    "ckpt_model_50000"
)
if [[ `hostname` == "sc.stanford.edu" ]]; then
    ENV_KWARGS="${ENV_KWARGS} --gui 0"
fi

for EXP_NAME in "${experiments[@]}"; do
    for ckpt in "${ckpts[@]}"; do
        for policy_env in "${policy_envs[@]}"; do
            POLICY_CHECKPOINT="models/${EXP_NAME}/${policy_env}/${ckpt}.pt"
            # ENV_CONFIG="configs/pybullet/envs/examples/primitives/${policy_env}_single_rack.yaml"
            # DEBUG_RESULTS="plots/${EXP_NAME}/${policy_env}/results_12.npz"
            eval_policies
        done
    done
done
