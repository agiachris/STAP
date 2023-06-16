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
        args="${args} --verbose 1"
    fi
    CMD="python scripts/eval/eval_planners.py ${args}"
    run_cmd
}

function run_planners {
    for task in "${TASKS[@]}"; do
        PLANNER_OUTPUT_PATH="${PLANNER_OUTPUT_ROOT}/${task}"
        ENV_CONFIG="${TASK_ROOT}/${task}.yaml"

        for planner in "${PLANNERS[@]}"; do
            PLANNER_CONFIG="${PLANNER_CONFIG_PATH}/${planner}.yaml"

            POLICY_CHECKPOINTS=()
            for primitive in "${PRIMITIVES[@]}"; do
                POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${primitive}/${CHECKPOINT}.pt")
            done

            SCOD_CHECKPOINTS=()

            if [[ "${planner}" == *_oracle_*dynamics ]]; then
                DYNAMICS_CHECKPOINT=""
            else
                DYNAMICS_CHECKPOINT="${DYNAMICS_INPUT_PATH}/${CHECKPOINT}.pt"
            fi
    
            eval_planner
        done
    done
}

function visualize_results {
    args=""
    args="${args} --path ${PLANNER_OUTPUT_ROOT}"
    args="${args} --envs ${TASKS[@]}"
    args="${args} --methods ${PLANNERS[@]}"
    if [ ! -z "${FIGURE_NAME}" ]; then
        args="${args} --name ${FIGURE_NAME}"
    fi
    CMD="python scripts/visualize/generate_planning_figure.py ${args}"
    run_cmd
}

# Evaluation tasks: Uncomment tasks to evaluate.
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

# Planners: Uncomment planners to evaluate.
PLANNER_CONFIG_PATH="configs/pybullet/planners"
PLANNERS=(
# Q-value / Learned dynamics.
    "policy_cem"
    "random_cem"
    "policy_shooting"
    "random_shooting"
# SCOD UQ value / Learned dynamics.
    # "policy_cem_var_scod_value"
    # "policy_cem_cvar_scod_value"
    # "policy_shooting_var_scod_value"
    # "policy_shooting_cvar_scod_value"
# Q-value / Oracle dynamics.
    # "policy_cem_oracle_dynamics"
    # "random_cem_oracle_dynamics"
    # "policy_shooting_oracle_dynamics"
    # "random_shooting_oracle_dynamics"
# SCOD UQ value / Oracle dynamics
    # "policy_cem_var_scod_value_oracle_dynamics"
    # "policy_shooting_var_scod_value_oracle_dynamics"
    # "policy_cem_cvar_scod_value_oracle_dynamics"
    # "policy_shooting_cvar_scod_value_oracle_dynamics"
# SCOD UQ thresholding / Latent Dynamics.
    # "scod_policy_shooting"
    # "scod_policy_cem"
# SCOD UQ thresholding / Oracle Dynamics.
    # "scod_policy_shooting_oracle_dynamics"
    # "scod_policy_cem_oracle_dynamics"
# Oracle value / Oracle dynamics.
    # "policy_cem_oracle_value_dynamics"
    # "random_cem_oracle_value_dynamics"
    # "policy_shooting_oracle_value_dynamics"
    # "random_shooting_oracle_value_dynamics"
# Greedy.
    # "greedy_oracle_dynamics"
    "greedy"
)

# Setup.
SBATCH_SLURM="scripts/eval/eval_planners_juno.sh"
DEBUG=0

input_path="models"
output_path="plots"

# Pybullet experiments.
if [[ `hostname` == *stanford.edu ]] || [[ `hostname` == juno* ]]; then
    ENV_KWARGS="--gui 0"
fi
ENV_KWARGS="${ENV_KWARGS} --closed-loop 1"

# Evaluate planners.
exp_name="planning"
PLANNER_OUTPUT_ROOT="${output_path}/${exp_name}"

PRIMITIVES=(
    "pick"
    "place"
    "pull"
    "push"
)
CHECKPOINT="official_model"
POLICY_INPUT_PATH="${input_path}/primitives_light_mse"
DYNAMICS_INPUT_PATH="${input_path}/dynamics/pick_place_pull_push_dynamics"
run_planners