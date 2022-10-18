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

function train_baseline {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --dynamics-config ${DYNAMICS_CONFIG}"
    args="${args} --agent-config ${AGENT_CONFIG}"
    args="${args} --env-config ${ENV_CONFIG}"
    # args="${args} --eval-env-config ${EVAL_ENV_CONFIG}"
    args="${args} --planner-config ${PLANNER_CONFIG}"
    if [ ! -z "${POLICY_CHECKPOINTS}" ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS}"
    fi
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path ${OUTPUT_PATH}_debug"
        # args="${args} --eval-recording-path ${EVAL_RECORDING_PATH}_debug"
        args="${args} --overwrite"
        args="${args} --num-pretrain-steps 100"
        args="${args} --num-train-steps 10"
        args="${args} --num-eval-steps 1"
    else
        args="${args} --path ${OUTPUT_PATH}"
        # args="${args} --eval-recording-path ${EVAL_RECORDING_PATH}"
    fi

    CMD="python scripts/train/train_baselines.py ${args}"
    run_cmd
}

# Setup.
DEBUG=0
output_path="models"
plots_path="plots"

# Experiments.

# exp_name="20220915/official"
# AGENT_CONFIG="configs/pybullet/agents/sac.yaml"

exp_name="20220915/official_lff"
AGENT_CONFIG="configs/pybullet/agents/sac_lff.yaml"

planners=(
    # "daf_policy_cem"
    # "daf_random_cem"
    # "daf_policy_shooting"
    # "daf_random_shooting"
    "dreamer_greedy"
)
envs=(
    # "hook_reach/task0"
    # "hook_reach/task1"
    "hook_reach/task2"
    # "constrained_packing/task0"
    # "constrained_packing/task1"
    "constrained_packing/task2"
    # "rearrangement_push/task0"
    # "rearrangement_push/task1"
    "rearrangement_push/task2"
)
# eval_envs=(
#     "hook_reach/task0"
#     # "hook_reach/task1"
#     # "hook_reach/task2"
#     # "hook_reach/task3"
#     # "hook_reach/task4"
#     # "constrained_packing/task0"
#     # "constrained_packing/task1"
#     # "constrained_packing/task2"
#     # "constrained_packing/task3"
#     # "constrained_packing/task4"
#     # "rearrangement_push/task0"
#     # "rearrangement_push/task1"
#     # "rearrangement_push/task2"
#     # "rearrangement_push/task3"
#     # "rearrangement_push/task4"
# )

# TRAINER_CONFIG="configs/pybullet/trainers/daf.yaml"
TRAINER_CONFIG="configs/pybullet/trainers/dreamer.yaml"

DYNAMICS_CONFIG="configs/pybullet/dynamics/table_env.yaml"
ENV_KWARGS="--num-env-processes 4 --num-eval-env-processes 2 --closed-loop-planning 1"
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]]; then
    ENV_KWARGS="--gui 0"
fi

num_envs=${#envs[@]}
for (( i=0; i<${num_envs}; i++ )); do
    env=${envs[$i]}
    eval_env=${eval_envs[$i]}
    ENV_CONFIG="configs/pybullet/envs/official/domains/${env}.yaml"
    EVAL_ENV_CONFIG="configs/pybullet/envs/official/domains/${eval_env}.yaml"
    for planner in "${planners[@]}"; do
        PLANNER_CONFIG="configs/pybullet/planners/${planner}.yaml"
        OUTPUT_PATH="${output_path}/${exp_name}/${env}/${planner}"

        train_baseline
    done
done
