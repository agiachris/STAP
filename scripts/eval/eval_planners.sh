#!/bin/bash

set -e

GCP_LOGIN="juno-login-lclbjqwy-001"

function run_cmd {
    echo ""
    echo "${CMD}"
    if [[ `hostname` == "sc.stanford.edu" ]]; then
        sbatch scripts/eval/eval_planners_juno.sh "${CMD}"
    elif [[ `hostname` == "${GCP_LOGIN}" ]]; then
        sbatch scripts/train/eval_gcp.sh "${CMD}"
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
            if [[ "${planner}" == daf_* ]]; then
                POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${planner}/${policy_env}/${CKPT}.pt")
            else
                POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${policy_env}/${CKPT}.pt")
            fi
        done

        SCOD_CHECKPOINTS=()
        if [[ "${planner}" == *scod_value* ]]; then
            for policy_env in "${POLICY_ENVS[@]}"; do
                SCOD_CHECKPOINTS+=("${SCOD_INPUT_PATH}/${CKPT}/${policy_env}/scod/final_scod.pt")
            done
        fi

        if [[ "${planner}" == *_oracle_*dynamics ]]; then
            DYNAMICS_CHECKPOINT=""
        elif [[ "${planner}" == daf_* ]]; then
            DYNAMICS_CHECKPOINT="${DYNAMICS_INPUT_PATH}/${planner}/dynamics/final_model.pt"
        else
            DYNAMICS_CHECKPOINT="${DYNAMICS_INPUT_PATH}/${CKPT}/dynamics/final_model.pt"
        fi

        eval_planner
    done
}

function visualize_results {
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
# Q-value / Latent dynamics.
    "policy_cem"
    "random_cem"
    "policy_shooting"
    "random_shooting"
# SCOD value / Latent dynamics.
    "policy_cem_var_scod_value"
    "policy_cem_cvar_scod_value"
    "policy_shooting_var_scod_value"
    "policy_shooting_cvar_scod_value"
# Q-value / Oracle dynamics.
    "policy_cem_oracle_dynamics"
    "random_cem_oracle_dynamics"
    "policy_shooting_oracle_dynamics"
    "random_shooting_oracle_dynamics"
    "policy_cem_var_scod_value_oracle_dynamics"
    "policy_shooting_var_scod_value_oracle_dynamics"
    "policy_cem_cvar_scod_value_oracle_dynamics"
    "policy_shooting_cvar_scod_value_oracle_dynamics"
# Oracle value / Oracle dynamics.
    "policy_cem_oracle_value_dynamics"
    "random_cem_oracle_value_dynamics"
    "policy_shooting_oracle_value_dynamics"
    "random_shooting_oracle_value_dynamics"
# DAF.
    "daf_policy_cem"
    "daf_policy_shooting"
    "daf_random_cem"
    "daf_random_shooting"
# Greedy.
    "greedy_oracle_dynamics"
    "greedy"
)

# Experiments.

# Pybox2d.
# exp_name="20220727/pybox2d"
# PLANNER_CONFIG_PATH="configs/pybox2d/planners"
# ENV_CONFIG_PATH="configs/pybox2d/envs"
# POLICY_ENVS=("placeright" "pushleft")
# checkpoints=(
#     "final_model"
#     "best_model"
#     "ckpt_model_50000"
# )

# Pybullet.
exp_name="20220905/official"
PLANNER_CONFIG_PATH="configs/pybullet/planners"
env_configs=(
    "configs/pybullet/envs/official/domains/hook_reach/task0.yaml"
    # "configs/pybullet/envs/official/domains/hook_reach/task1.yaml"
    # "configs/pybullet/envs/official/domains/hook_reach/task2.yaml"
    # "configs/pybullet/envs/official/domains/hook_reach/task3.yaml"
    # "configs/pybullet/envs/official/domains/hook_reach/task4.yaml"
)
POLICY_ENVS=("pick" "place" "pull" "push")
checkpoints=(
    # "final_model"
    # "best_model"
    "ckpt_model_50000"
)
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]]; then
    ENV_KWARGS="--gui 0"
fi

# Run planners.
POLICY_INPUT_PATH="${input_path}/${exp_name}"
SCOD_INPUT_PATH="${input_path}/${exp_name}"
DYNAMICS_INPUT_PATH="${input_path}/${exp_name}"
for CKPT in "${checkpoints[@]}"; do
    for ENV_CONFIG in "${env_configs[@]}"; do
        env_name=${ENV_CONFIG//.yaml/}  # Cut off .yaml.
        env_name=(${env_name//\// })  # Tokenize by '/'.
        env_name=${env_name[@]: -2}  # Extract last two tokens.
        env_name="${env_name// /\/}"  # Add back '/'.
        PLANNER_OUTPUT_PATH="${output_path}/${exp_name}/${CKPT}/${env_name}"
        run_planners
    done
done

# Visualize results.
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]] || [ $DEBUG -ne 0 ]; then
    exit
fi

for ckpt in "${checkpoints[@]}"; do
    PLANNER_OUTPUT_PATH="${output_path}/${exp_name}/${ckpt}"
    visualize_results
done
