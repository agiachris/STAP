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
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path plots/${EXP_NAME}_debug"
        args="${args} --verbose 1"
    else
        args="${args} --path plots/${EXP_NAME}"
        args="${args} --verbose 0"
    fi
    CMD="python scripts/eval/eval_table_env_policies.py ${args}"
    run_cmd
}

# Setup.

DEBUG=0

# Evaluate policies.

policy_envs=(
    "pick"
)
experiments=(
    "20220705/pick_box"
    # "20220708/pick_hook"
)

for EXP_NAME in "${experiments[@]}"; do
    for policy_env in "${policy_envs[@]}"; do
        POLICY_CHECKPOINT="models/${EXP_NAME}/${policy_env}/final_model.pt"
        eval_policies
    done
done
