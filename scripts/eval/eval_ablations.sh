#!/bin/bash

set -e

# GCP_LOGIN="juno-login-lclbjqwy-001"
GCP_LOGIN="gcp-login-yq0fvtuw-001"

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

function eval_planner {
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
    if [[ ! -z "${LOAD_PATH}" ]]; then
        args="${args} --load-path ${LOAD_PATH}"
    fi
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --num-eval 1"
        args="${args} --path ${PLANNER_OUTPUT_PATH}_debug"
        args="${args} --verbose 1"
    else
        args="${args} --num-eval 100"
        args="${args} --path ${PLANNER_OUTPUT_PATH}"
        args="${args} --verbose 0"
    fi
    CMD="python scripts/eval/eval_planners.py ${args}"
    run_cmd
}

function run_planners {
    for planner in "${PLANNERS[@]}"; do
        PLANNER_CONFIG="${PLANNER_CONFIG_PATH}/${planner}.yaml"

        POLICY_CHECKPOINTS=()
        for policy_env in "${POLICY_ENVS[@]}"; do
            POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${policy_env}/${CKPT}.pt")
        done

        SCOD_CHECKPOINTS=()
        if [[ "${planner}" == *scod* ]]; then
            for policy_env in "${POLICY_ENVS[@]}"; do
                SCOD_CHECKPOINTS+=("${SCOD_INPUT_PATH}/${CKPT}/${policy_env}/${SCOD_CONFIG}/final_scod.pt")
            done
        fi

        if [[ "${planner}" == *_oracle_*dynamics ]]; then
            DYNAMICS_CHECKPOINT=""
        else
            DYNAMICS_CHECKPOINT="${DYNAMICS_INPUT_PATH}/${CKPT}/dynamics/final_model.pt"
        fi

        eval_planner
    done
}

function visualize_planners {
    args=""
    args="${args} --path ${PLANNER_OUTPUT_PATH}"
    args="${args} --envs ${ENVS[@]}"
    args="${args} --methods ${PLANNERS[@]}"
    CMD="python scripts/visualize/visualize_planners.py ${args}"
    run_cmd
}

# Setup.
DEBUG=0
input_path="models"
output_path="plots"

# Evaluate planners.
PLANNERS=(
# Oracles.
    # "policy_shooting_oracle_value_dynamics"
    # "scod_policy_cem_oracle_dynamics"
    # "policy_cem_oracle_dynamics"
# Planning w/ SCOD.
    "scod_policy_cem"
    "scod_policy_cem_linear"
    "scod_policy_cem_geometric"
    "policy_cem_var_scod_value"
    # "policy_cem_cvar_scod_value"
# Planning w/o SCOD.
    "policy_cem"
    # "random_cem"
    # "policy_shooting"
    # "random_shooting"
    # "greedy"
)

# Experiments.

# Pybullet.
exp_name="20220912/official"
PLANNER_CONFIG_PATH="configs/pybullet/planners/"
ENVS=(
    # "hook_reach/task0"
    "hook_reach/task1"
    "hook_reach/task2"
    # "constrained_packing/task0"
    "constrained_packing/task1"
    "constrained_packing/task2"
    # "rearrangement_push/task0"
    "rearrangement_push/task1"
    "rearrangement_push/task2"
)
POLICY_ENVS=("pick" "place" "pull" "push")
CKPT="selectscod_model"
SCOD_CONFIG="scod"
ENV_KWARGS="--closed-loop 1"
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]]; then
    ENV_KWARGS="--gui 0"
fi

# Run planners.
POLICY_INPUT_PATH="${input_path}/${exp_name}"
SCOD_INPUT_PATH="${input_path}/${exp_name}"
DYNAMICS_INPUT_PATH="${input_path}/${exp_name}"
for env in "${ENVS[@]}"; do
    ENV_CONFIG="configs/pybullet/envs/official/domains/${env}.yaml"
    PLANNER_OUTPUT_PATH="${output_path}/${exp_name}/ablation_experiment/${env}"
    run_planners
done

# Visualize results.
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]] || [ $DEBUG -ne 0 ]; then
    exit
fi


PLANNER_OUTPUT_PATH="${output_path}/${exp_name}/ablation_experiment"
visualize_planners
