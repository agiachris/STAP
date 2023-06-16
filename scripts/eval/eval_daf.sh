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
        for task_idx in 0 1 2; do
            eval_task="${task::-1}${task_idx}"
            
            ENV_CONFIG="${TASK_ROOT}/${eval_task}.yaml"
            PLANNER_OUTPUT_PATH="${PLANNER_OUTPUT_ROOT}/${eval_task}/train${task: -1}"
            
            for planner in "${PLANNERS[@]}"; do
                PLANNER_CONFIG="${PLANNER_CONFIG_PATH}/${planner}.yaml"                

                SCOD_CHECKPOINTS=()
                POLICY_CHECKPOINTS=()
                for primitive in "${PRIMITIVES[@]}"; do
                    POLICY_CHECKPOINTS+=("${DAF_CHECKPOINT_ROOT}/${task}/${DAF_CHECKPOINT_PLANNER}/${POLICY_CHECKPOINT_PATHS[${primitive}]}")  
                done

                DYNAMICS_CHECKPOINT="${DAF_CHECKPOINT_ROOT}/${task}/${DAF_CHECKPOINT_PLANNER}/${DYNAMICS_CHECKPOINT_PATH}"
        
                eval_planner
            done
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

# Planners.
PLANNERS=(
# DAF.
    # "daf_policy_cem"
    # "daf_policy_shooting"
    "daf_random_cem_light"
    # "daf_random_cem" 
    # "daf_random_shooting"
)

# Evaluation tasks.
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
SBATCH_SLURM="scripts/eval/eval_planners_juno.sh"
DEBUG=0

input_path="models"
output_path="plots"

# Pybullet experiments.
if [[ `hostname` == *stanford.edu ]] || [[ `hostname` == juno* ]]; then
    ENV_KWARGS="--gui 0"
fi
ENV_KWARGS="${ENV_KWARGS} --closed-loop 1"

exp_name="planning"
PLANNER_OUTPUT_ROOT="${output_path}/${exp_name}"
PLANNER_CONFIG_PATH="configs/pybullet/planners/baselines"

# Evaluation Deep Affordance Foresight (DAF-Skills).
PRIMITIVES=(
    "pick"
    "place"
    "pull"
    "push"
)
DAF_CHECKPOINT_ROOT="${input_path}/baselines"
DAF_CHECKPOINT_PLANNER="daf_random_cem_light"
declare -A POLICY_CHECKPOINT_PATHS=(
    ["pick"]="pick/best_model.pt"
    ["place"]="place/best_model.pt"
    ["pull"]="pull/best_model.pt"
    ["push"]="push/best_model.pt"
)
DYNAMICS_CHECKPOINT_PATH="dynamics/best_model.pt"
run_planners