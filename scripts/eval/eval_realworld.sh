#!/bin/bash

set -e

function eval_realworld {
    args=""
    args="${args} --planner-config ${PLANNER_CONFIG}"
    args="${args} --env-config ${ENV_CONFIG}"
    if [ ${#POLICY_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS[@]}"
    fi
    if [ ${#SCOD_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --scod-checkpoints ${SCOD_CHECKPOINTS[@]}"
    fi
    if [ ! -z "${DYNAMICS_CHECKPOINT}" ]; then
        args="${args} --dynamics-checkpoint ${DYNAMICS_CHECKPOINT}"
    fi
    if [[ ! -z "${LOAD_NPZ}" ]]; then
        args="${args} --load-npz ${LOAD_NPZ}"
    fi
    args="${args} ${ENV_KWARGS}"
    args="${args} --path ${PLANNER_OUTPUT_PATH}"
    CMD="python scripts/eval/eval_realworld.py ${args}"
    ${CMD}
}

function run_planners {
    PLANNER_CONFIG="${PLANNER_CONFIG_PATH}/${PLANNER}.yaml"

    POLICY_CHECKPOINTS=()
    for primitive in "${PRIMITIVE[@]}"; do
        POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${primitive}/${CHECKPOINT}.pt")
    done

    SCOD_CHECKPOINTS=()

    if [[ "${PLANNER}" == *_oracle_*dynamics ]]; then
        DYNAMICS_CHECKPOINT=""
    else
        DYNAMICS_CHECKPOINT="${DYNAMICS_INPUT_PATH}/${CHECKPOINT}.pt"
    fi

    eval_realworld
}

# Setup.
input_path="models"
output_path="plots"

# Planners: Uncomment planners to evaluate.
PLANNER_CONFIG_PATH="configs/pybullet/planners"
# PLANNER="policy_cem"
# PLANNER="greedy"

# Evaluation tasks: Uncomment tasks to evaluate.
TASK_ROOT="configs/pybullet/envs/official/real_domains"
# TASK="hook_reach/task2"
# TASK="constrained_packing/task1"
# TASK="rearrangement_push/task0"

# Pybullet experiments.
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]]; then
    ENV_KWARGS="--gui 0"
fi
ENV_KWARGS="${ENV_KWARGS} --closed-loop 1"

# Evaluate planners.
exp_name="real_world"
PLANNER_OUTPUT_PATH="${output_path}/${exp_name}/${TASK}/${PLANNER}"

PRIMITIVES=(
    "pick"
    "place"
    "pull"
    "push"
)
CHECKPOINT="official_model"
POLICY_INPUT_PATH="${input_path}/primitives_light_mse"
DYNAMICS_INPUT_PATH="${input_path}/dynamics/pick_place_pull_push_dynamics"
ENV_CONFIG="${TASK_ROOT}/${TASK}"
run_planners