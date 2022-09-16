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

function eval_tamp {
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
    args="${args} --seed 0"
    args="${args} --pddl-domain ${PDDL_DOMAIN}"
    args="${args} --pddl-problem ${PDDL_PROBLEM}"
    args="${args} --max-depth 2"
    args="${args} --timeout 10"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --num-eval 10"
        args="${args} --path ${PLANNER_OUTPUT_PATH}_debug"
        args="${args} --verbose 1"
    else
        args="${args} --num-eval 100"
        args="${args} --path ${PLANNER_OUTPUT_PATH}"
        args="${args} --verbose 0"
    fi
    CMD="python scripts/eval/eval_tamp.py ${args}"
    run_cmd
}

function run_planners {
    for planner in "${PLANNERS[@]}"; do
        PLANNER_CONFIG="${PLANNER_CONFIG_PATH}/${planner}.yaml"

        POLICY_CHECKPOINTS=()
        for policy_env in "${POLICY_ENVS[@]}"; do
            if [[ "${planner}" == daf_* ]]; then
                POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${planner}/${policy_env}/${CKPT}.pt")
            else
                POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${policy_env}/${CKPT}.pt")
            fi
        done

        SCOD_CHECKPOINTS=()
        if [[ "${planner}" == *scod* ]]; then
            for policy_env in "${POLICY_ENVS[@]}"; do
                SCOD_CHECKPOINTS+=("${SCOD_INPUT_PATH}/${CKPT}/${policy_env}/${SCOD_CONFIG}/final_scod.pt")
            done
        fi

        if [[ "${planner}" == *_oracle_*dynamics ]]; then
            DYNAMICS_CHECKPOINT=""
        elif [[ "${planner}" == daf_* ]]; then
            DYNAMICS_CHECKPOINT="${DYNAMICS_INPUT_PATH}/${planner}/dynamics/final_model.pt"
        else
            DYNAMICS_CHECKPOINT="${DYNAMICS_INPUT_PATH}/${CKPT}/dynamics/final_model.pt"
        fi

        eval_tamp
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

# Evaluate planners.
PLANNERS=(
    "ablation/policy_cem"
    "ablation/scod_policy_cem"
    "ablation/policy_shooting"
    # "daf_random_shooting"
    "ablation/random_cem"
    "ablation/random_shooting"
    "greedy"
)

# Experiments.

# Pybullet.
exp_name="20220914/official"
PLANNER_CONFIG_PATH="configs/pybullet/planners"
ENVS=(
    # "hook_reach/tamp0"
    "constrained_packing/tamp0"
    # "rearrangement_push/tamp0"
)
POLICY_ENVS=("pick" "place" "pull" "push")
CKPT="select_model"
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
    PDDL_DOMAIN="configs/pybullet/envs/official/domains/${env}_domain.pddl"
    PDDL_PROBLEM="configs/pybullet/envs/official/domains/${env}_problem.pddl"
    PLANNER_OUTPUT_PATH="${output_path}/${exp_name}/tamp_experiment/${env}"
    run_planners
done

# Visualize results.
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]] || [ $DEBUG -ne 0 ]; then
    exit
fi

PLANNER_OUTPUT_PATH="${output_path}/${exp_name}/tamp_experiment"
visualize_tamp
