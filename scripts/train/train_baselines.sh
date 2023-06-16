#!/bin/bash

set -e

function run_cmd {
    echo ""
    echo "${CMD}"
    if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == juno* ]]; then
        sbatch "${SBATCH_SLURM}" "${CMD}"
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
    args="${args} --planner-config ${PLANNER_CONFIG}"
    if [ ! -z "${POLICY_CHECKPOINTS}" ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS}"
    fi
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path ${PLANNER_OUTPUT_PATH}_debug"
        args="${args} --overwrite"
        args="${args} --num-pretrain-steps 100"
        args="${args} --num-train-steps 10"
        args="${args} --num-eval-steps 1"
    else
        args="${args} --path ${PLANNER_OUTPUT_PATH}"
    fi

    CMD="python scripts/train/train_baselines.py ${args}"
    run_cmd
}


function run_baselines {
    for task in "${TASKS[@]}"; do
        ENV_CONFIG="${TASK_ROOT}/${task}.yaml"

        for planner in "${PLANNERS[@]}"; do
            PLANNER_OUTPUT_PATH="${PLANNER_OUTPUT_ROOT}/${task}/${planner}"
            PLANNER_CONFIG="${PLANNER_CONFIG_PATH}/${planner}.yaml"
            train_baseline
        done
    done
}

TASK_ROOT="configs/pybullet/envs/official/sim_domains"
TASKS=(
# Domain 1: Hook Reach
    "hook_reach/task0"
    "hook_reach/task1"
    "hook_reach/task2"
# Domain 2: Constrained Packing
    "constrained_packing/task0"
    "constrained_packing/task1"
    "constrained_packing/task2"
# Domain 3: Rearrangement Push
    "rearrangement_push/task0"
    "rearrangement_push/task1"
    "rearrangement_push/task2"
)

# Setup.
SBATCH_SLURM="scripts/train/train_juno.sh"
DEBUG=0

output_path="models"
plots_path="plots"

# Pybullet experiments.
if [[ `hostname` == *stanford.edu ]] || [[ `hostname` == juno* ]]; then
    ENV_KWARGS="--gui 0"
fi
ENV_KWARGS="${ENV_KWARGS} --closed-loop-planning 1"

exp_name="baselines"
PLANNER_OUTPUT_ROOT="${output_path}/${exp_name}"
PLANNER_CONFIG_PATH="configs/pybullet/planners/baselines"

# Train Deep Affordance Foresight (DAF-Skills).
AGENT_CONFIG="configs/pybullet/agents/single_stage/sac.yaml"
TRAINER_CONFIG="configs/pybullet/trainers/baselines/daf.yaml"
DYNAMICS_CONFIG="configs/pybullet/dynamics/table_env.yaml"

PLANNERS=(
    # "daf_policy_cem"
    # "daf_random_cem"
    "daf_random_cem_light"
    # "daf_policy_shooting"
    # "daf_random_shooting"
)
run_baselines

# # Train Dreamer (not passing).
# AGENT_CONFIG="configs/pybullet/agents/single_stage/sac.yaml"
# TRAINER_CONFIG="configs/pybullet/trainers/baselines/dreamer.yaml"
# DYNAMICS_CONFIG="configs/pybullet/dynamics/table_env.yaml"

# PLANNERS=(
#     # "dreamer_greedy"
# )
# run_baselines
