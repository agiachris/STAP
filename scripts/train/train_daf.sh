#!/bin/bash

set -e

GCP_LOGIN="juno-login-lclbjqwy-001"

function run_cmd {
    echo ""
    echo "${CMD}"
    if [[ `hostname` == "sc.stanford.edu" ]]; then
        sbatch scripts/train/train_juno.sh "${CMD}"
    elif [[ `hostname` == "${GCP_LOGIN}" ]]; then
        sbatch scripts/train/train_gcp.sh "${CMD}"
    else
        ${CMD}
    fi
}

function train_daf {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --dynamics-config ${DYNAMICS_CONFIG}"
    args="${args} --agent-config ${AGENT_CONFIG}"
    args="${args} --env-config ${ENV_CONFIG}"
    args="${args} --planner-config ${PLANNER_CONFIG}"
    if [ ! -z "${POLICY_CHECKPOINTS}" ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS}"
    fi
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path ${OUTPUT_PATH}_debug"
        args="${args} --eval-recording-path ${EVAL_RECORDING_PATH}_debug"
        args="${args} --overwrite"
        args="${args} --num-pretrain-steps 10"
        args="${args} --num-train-steps 10"
        args="${args} --num-eval-steps 10"
    else
        args="${args} --path ${OUTPUT_PATH}"
        args="${args} --eval-recording-path ${EVAL_RECORDING_PATH}"
    fi

    CMD="python scripts/train/train_daf.py ${args}"
    run_cmd
}

# Setup.
DEBUG=0
output_path="models"
plots_path="plots"

# Experiments.

exp_name="20220907/official"

planners=(
    # "daf_policy_cem"
    "daf_random_cem"
    # "daf_policy_shooting"
    # "daf_random_shooting"
)
envs=(
    "hook_reach/task0"
    "hook_reach/task1"
    "hook_reach/task2"
    "hook_reach/task3"
    "hook_reach/task4"
    "constrained_packing/task0"
    "constrained_packing/task1"
    "constrained_packing/task2"
    "constrained_packing/task3"
    "constrained_packing/task4"
    "rearrangement_push/task0"
    "rearrangement_push/task1"
    "rearrangement_push/task2"
    "rearrangement_push/task3"
    "rearrangement_push/task4"
)

TRAINER_CONFIG="configs/pybullet/trainers/daf.yaml"
DYNAMICS_CONFIG="configs/pybullet/dynamics/table_env.yaml"
AGENT_CONFIG="configs/pybullet/agents/sac.yaml"
ENV_KWARGS="--num-env-processes 6"
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]]; then
    ENV_KWARGS="--gui 0"
fi

for env in "${envs[@]}"; do
    ENV_CONFIG="configs/pybullet/envs/official/domains/${env}.yaml"
    for planner in "${planners[@]}"; do
        PLANNER_CONFIG="configs/pybullet/planners/${planner}.yaml"
        OUTPUT_PATH="${output_path}/${exp_name}/${env}/${planner}"
        EVAL_RECORDING_PATH="${plots_path}/${exp_name}/${planner}"

        train_daf
    done
done
