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
    CMD="python scripts/eval/eval_policies.py ${args}"
    run_cmd
}

# Setup.

DEBUG=0
NUM_EPISODES=10

# Evaluate policies.

policy_envs=(
    # "pick"
    "pick_0"
    # "place_0"
    # "pull_0"
    # "push_0"
)
# experiments=(
#     "official"
# )
# experiments=(
#     "20230101/complete_q_multistage"
# )
experiments=(
    "20230106/complete_q_multistage"
)
ckpts=(
    # "best_model"
    # "final_model"
    # "select_model"
    # "ckpt_model_550000"
    # "ckpt_model_100000"
    # "ckpt_model_150000"
    # "ckpt_model_300000"
    "ckpt_model_1000000"
)
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]]; then
    ENV_KWARGS="${ENV_KWARGS} --gui 0"
fi

ENV_KWARGS="${ENV_KWARGS} --gui 0"

for exp_name in "${experiments[@]}"; do
    for ckpt in "${ckpts[@]}"; do
        for policy_env in "${policy_envs[@]}"; do
            # set ENV_CONFIG to configs/pybullet/envs/official/primitives/<policy_env_name_without_0>_eval.yaml
            # ENV_CONFIG="configs/pybullet/envs/official/primitives/${policy_env:0:-2}_eval.yaml"
            # ENV_CONFIG="configs/pybullet/envs/official/primitives/${policy_env:0:-2}_viz.yaml"
            ENV_CONFIG="configs/pybullet/envs/official/domains/hook_reach/task0.yaml"
            EXP_NAME="${exp_name}/${ckpt}"
            POLICY_CHECKPOINT="models/${exp_name}/${policy_env}/${ckpt}.pt"
            eval_policies
        done
    done
done
