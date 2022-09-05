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
    if [[ -n "${EVAL_RESULTS}" ]]; then
        args="${args} --eval-results ${EVAL_RESULTS}"
    fi
    if [[ -n "${ENV_CONFIG}" ]]; then
        args="${args} --env-config ${ENV_CONFIG}"
    fi
    CMD="python scripts/eval/eval_table_env_policies.py ${args}"
    run_cmd
}

# Setup.

DEBUG=0

# Evaluate policies.

policy_envs=(
    # "pick"
    "place"
)
experiments=(
    "20220903/examples_collisions"
    "20220904/examples_curriculum"
    "20220904/examples_pretrain5000"
    "20220904/examples_mlp3layer"
)

for EXP_NAME in "${experiments[@]}"; do
    for policy_env in "${policy_envs[@]}"; do
        POLICY_CHECKPOINT="models/${EXP_NAME}/${policy_env}/final_model.pt"
        ENV_CONFIG="configs/pybullet/envs/examples/primitives/${policy_env}_single_rack.yaml"
        # EVAL_RESULTS="plots/${EXP_NAME}/${policy_env}/results_12.npz"
        eval_policies
    done
done
