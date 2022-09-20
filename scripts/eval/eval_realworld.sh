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
    for policy_env in "${POLICY_ENVS[@]}"; do
        if [[ "${PLANNER}" == daf_* ]]; then
            POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${PLANNER}/${policy_env}/${CKPT}.pt")
        else
            POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${policy_env}/${CKPT}.pt")
        fi
    done

    SCOD_CHECKPOINTS=()
    if [[ "${PLANNER}" == *scod* ]]; then
        for policy_env in "${POLICY_ENVS[@]}"; do
            SCOD_CHECKPOINTS+=("${SCOD_INPUT_PATH}/${CKPT}/${policy_env}/scod/final_scod.pt")
        done
    fi

    if [[ "${PLANNER}" == *_oracle_*dynamics ]]; then
        DYNAMICS_CHECKPOINT=""
    elif [[ "${PLANNER}" == daf_* ]]; then
        DYNAMICS_CHECKPOINT="${DYNAMICS_INPUT_PATH}/${PLANNER}/dynamics/final_model.pt"
    else
        DYNAMICS_CHECKPOINT="${DYNAMICS_INPUT_PATH}/${CKPT}/dynamics/final_model.pt"
    fi

    eval_realworld
}

# Setup.
input_path="models"
output_path="plots"

# Evaluate planners.
# PLANNER="policy_cem"
# PLANNER="greedy"
# ENV="hook_reach/task2"
# ENV="constrained_packing/task1"
# ENV="rearrangement_push/task0"

# Experiments.
exp_name="20220914/official"
PLANNER_CONFIG_PATH="configs/pybullet/planners"
POLICY_ENVS=("pick" "place" "pull" "push")
CKPT="select_model"
ENV_KWARGS="--closed-loop 1"
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]]; then
    ENV_KWARGS="--gui 0"
fi

# Run planners.
POLICY_INPUT_PATH="${input_path}/${exp_name}"
SCOD_INPUT_PATH="${input_path}/${exp_name}"
DYNAMICS_INPUT_PATH="${input_path}/${exp_name}"

ENV_CONFIG="configs/pybullet/envs/real_world/domains/${ENV}.yaml"
PLANNER_OUTPUT_PATH="${output_path}/${exp_name}/${CKPT}/${ENV}"
LOAD_PATH="${PLANNER_OUTPUT_PATH}"
run_planners
