#!/bin/bash

set -e

# GCP_LOGIN="juno-login-lclbjqwy-001"
# GCP_LOGIN="gcp-login-yq0fvtuw-001"

function run_cmd {
    echo ""
    echo "${CMD}"
    if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == juno* ]]; then
        sbatch "${SBATCH_SLURM}" "${CMD}"
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
                POLICY_CHECKPOINTS+=("${POLICY_CHECKPOINT_PATHS[${primitive}]}")
            done

            SCOD_CHECKPOINTS=()

            if [[ "${planner}" == *_oracle_*dynamics ]]; then
                DYNAMICS_CHECKPOINT=""
            else
                DYNAMICS_CHECKPOINT="${DYNAMICS_CHECKPOINT_PATH}"
            fi
    
            eval_planner
        done
    done
}

### Setup.
SBATCH_SLURM="scripts/eval/eval_planners_juno.sh"
DEBUG=0
input_path="models"
output_path="plots"

### Experiments.

## Planner configurations.
PLANNERS=(
# Q-value / Latent dynamics.
    # "policy_cem"
    # "random_cem"
    # "policy_shooting"
    # "random_shooting"
# Ensemble Q-value / Latent dynamics.
    # "ensemble_policy_cem"
    # "ensemble_policy_cem_ood"
    # "ensemble_policy_cem_scale-0.1"
    # "ensemble_policy_cem_scale-0.5"
    # "ensemble_policy_cem_scale-1.0"
    # "ensemble_policy_cem_scale-0.1_pessimistic-True"
    # "ensemble_policy_cem_scale-0.5_pessimistic-True"
    # "ensemble_policy_cem_scale-1.0_pessimistic-True"
# SCOD value / Latent dynamics.
    # "policy_cem_var_scod_value"
    # "policy_cem_cvar_scod_value"
    # "policy_shooting_var_scod_value"
    # "policy_shooting_cvar_scod_value"
# Q-value / Oracle dynamics.
    # "policy_cem_oracle_dynamics"
    # "random_cem_oracle_dynamics"
    # "policy_shooting_oracle_dynamics"
    # "random_shooting_oracle_dynamics"
# SCOD value / Oracle dynamics
    # "policy_cem_var_scod_value_oracle_dynamics"
    # "policy_shooting_var_scod_value_oracle_dynamics"
    # "policy_cem_cvar_scod_value_oracle_dynamics"
    # "policy_shooting_cvar_scod_value_oracle_dynamics"
# SCOD thresholding / Latent Dynamics.
    # "scod_policy_shooting"
    # "scod_policy_cem"
# SCOD thresholding / Oracle Dynamics.
    # "scod_policy_shooting_oracle_dynamics"
    # "scod_policy_cem_oracle_dynamics"
# Oracle value / Oracle dynamics.
    # "policy_cem_oracle_value_dynamics"
    # "random_cem_oracle_value_dynamics"
    # "policy_shooting_oracle_value_dynamics"
    # "random_shooting_oracle_value_dynamics"
# DAF.
    # "daf_policy_cem"
    # "daf_policy_shooting"
    # "daf_random_cem"
    # "daf_random_shooting"
# Greedy.
    # "greedy_oracle_dynamics"
    # "greedy"
)
# PLANNERS=(
# # Q-value / Latent dynamics.
#     "policy_cem_iter-10_samples-100"
#     "policy_cem_iter-10_samples-10k"
#     "policy_cem_iter-10_samples-1k"
#     "policy_cem_iter-5_samples-100"
#     "policy_cem_iter-5_samples-10k"
#     "policy_cem_iter-5_samples-1k"
# )

## TAPS Evaluation tasks.

TASK_ROOT="configs/pybullet/envs/taps/official/domains"
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

## T2M Evaluation tasks.
# TASK_ROOT="configs/pybullet/envs/t2m/official/tasks"
# TASKS=(
#     "task0"
#     "task1"
#     "task2"
#     "task3"
#     "task4"
#     "task5"
#     "task6"
# )

## T2M Evaluation tasks.
# TASK_ROOT="configs/pybullet/envs/t2m/examples/figures"
# TASKS=(
#     "teaser"
#     "teaser_hierarchical"
# )

## Pybullet.
exp_name="20230315/planners/t2m"
PLANNER_OUTPUT_ROOT="${output_path}/${exp_name}"

PLANNER_CONFIG_PATH="configs/pybullet/planners/policy_cem_ablations"
ENV_KWARGS="--closed-loop 1"
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]] || [[ `hostname` == juno* ]]; then
    ENV_KWARGS="--gui 0"
fi

# Model suite.
PRIMITIVES=(
    "pick"
    "place"
    "pull"
    "push"
)

# Critics trained with MSE loss and no sigmoid activation, balanced data (40%).
declare -A POLICY_CHECKPOINT_PATHS=(
    ["pick"]="models/20230309/policy/pick/final_model/final_model.pt"
    ["place"]="models/20230313/policy/place/final_model/final_model.pt"
    ["pull"]="models/20230313/policy/pull/final_model/final_model.pt"
    ["push"]="models/20230313/policy/push/final_model/final_model.pt"
)
DYNAMICS_CHECKPOINT_PATH="models/20230313/dynamics/pick_place_pull_push_dynamics/best_model.pt"
run_planners


## Visualize results.
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]] || [[ `hostname` == juno* ]] || [ $DEBUG -ne 0 ]; then
    exit
fi

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

FIGURE_NAME="policy_cem_ablation"
visualize_results
