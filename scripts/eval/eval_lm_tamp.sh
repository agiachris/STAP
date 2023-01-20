#!/bin/bash

# Usage: PYTHONPATH=. bash scripts/eval/eval_lm_tamp.sh

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

function eval_lm_tamp {
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
    args="${args} --key-name ${KEY_NAME}"
    args="${args} --seed 0"
    args="${args} --pddl-domain ${PDDL_DOMAIN}"
    args="${args} --pddl-problem ${PDDL_PROBLEM}"
    args="${args} --timeout 10"
    args="${args} --n-examples ${N_INCONTEXT_EXAMPLES}"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --num-eval 3"
        args="${args} --path ${PLANNER_OUTPUT_PATH}_debug"
        args="${args} --verbose 1"
        args="${args} --engine curie"
    else
        args="${args} --num-eval 3"
        args="${args} --path ${PLANNER_OUTPUT_PATH}"
        args="${args} --verbose 0"
        args="${args} --engine davinci"
    fi
    CMD="python scripts/eval/eval_lm_tamp.py ${args}"
    run_cmd
}

function run_planners {
    for planner in "${PLANNERS[@]}"; do
        PLANNER_CONFIG="${PLANNER_CONFIG_PATH}/${planner}.yaml"

        POLICY_CHECKPOINTS=(
            "models/20230106/complete_q_multistage/pick_0/ckpt_model_1000000.pt"
            "models/20230101/complete_q_multistage/place_0/best_model.pt"
            "models/20230101/complete_q_multistage/pull_0/best_model.pt"
            "models/20230101/complete_q_multistage/push_0/best_model.pt"
        )
        if [[ "${planner}" == *_oracle_*dynamics ]]; then
            DYNAMICS_CHECKPOINT=""
        elif [[ "${planner}" == daf_* ]]; then
            DYNAMICS_CHECKPOINT="${DYNAMICS_INPUT_PATH}/${planner}/dynamics/final_model.pt"
        else
            DYNAMICS_CHECKPOINT="models/official/select_model/dynamics/best_model.pt"
        fi

        eval_lm_tamp
    done
}

function visualize_tamp {
    args=""
    args="${args} --path ${PLANNER_OUTPUT_PATH}"
    args="${args} --methods ${PLANNERS[@]}"
    CMD="python scripts/visualize/visualize_planners.py ${args}"
    run_cmd
}

# Setup.
DEBUG=0
input_path="models"
output_path="plots"
exp_name="20230118/lm_tamp"

# LLM
KEY_NAME="personal-all"
N_INCONTEXT_EXAMPLES=5

# Evaluate planners.
PLANNERS=(
    "policy_cem"
    # "greedy"
)

# Experiments.

PLANNER_CONFIG_PATH="configs/pybullet/planners"
TASK_NUMS=(
    "0"
    "1"
    "2"
    "3"
    "4"
    "5"
    "6"
)
POLICY_ENVS=("pick" "place" "pull" "push")
# CKPT="select_model"
# CKPT="ckpt_model_10"
CKPT="best_model"

SCOD_CONFIG="scod"
ENV_KWARGS="--closed-loop 1"
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]]; then
    ENV_KWARGS="--gui 0"
fi

# Run planners.
POLICY_INPUT_PATH="${input_path}/${exp_name}"
for task_num in "${TASK_NUMS[@]}"; do
    ENV_CONFIG="configs/pybullet/envs/t2m/official/tasks/task${task_num}.yaml"
    PDDL_DOMAIN="configs/pybullet/envs/t2m/official/tasks/symbolic_domain.pddl"
    PDDL_PROBLEM="configs/pybullet/envs/t2m/official/tasks/task${task_num}_symbolic.pddl"
    PLANNER_OUTPUT_PATH="${output_path}/${exp_name}"
    run_planners
done


# Visualize results.
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]] || [ $DEBUG -ne 0 ]; then
    exit
fi

PLANNER_OUTPUT_PATH="${output_path}/${exp_name}/tamp_experiment"
visualize_tamp
