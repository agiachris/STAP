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

        eval_planner
    done
}

function visualize_results {
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
# Q-value / Latent dynamics.
    # "policy_cem"
    # "random_cem"
    # "policy_shooting"
    # "random_shooting"
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
exp_name="20220914/official"
PLANNER_CONFIG_PATH="configs/pybullet/planners"
ENVS=(
    ## Domain 1: Hook Reach
    # "hook_reach/task0"
    # "hook_reach/task1"
    # "hook_reach/task2"
    ## Domain 2: Constrained Packing
    # "constrained_packing/task0"
    # "constrained_packing/task1"
    # "constrained_packing/task2"
    ## Domain 3: Rearrangement Push
    # "rearrangement_push/task0"
    # "rearrangement_push/task1"
    # "rearrangement_push/task2"
)
POLICY_ENVS=("pick" "place" "pull" "push")
checkpoints=(
    # "final_model"
    # "best_model"
    "select_model"
    # "ckpt_model_50000"
    # "ckpt_model_100000"
    # "ckpt_model_150000"
    # "ckpt_model_200000"
    # "select_model"
    # "selectscod_model"
)
ENV_KWARGS="--closed-loop 1"
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]]; then
    ENV_KWARGS="--gui 0"
fi

# Run planners.
POLICY_INPUT_PATH="${input_path}/${exp_name}"
SCOD_INPUT_PATH="${input_path}/${exp_name}"
SCOD_CONFIG="scod"
DYNAMICS_INPUT_PATH="${input_path}/${exp_name}"
for CKPT in "${checkpoints[@]}"; do
    for env in "${ENVS[@]}"; do
        ENV_CONFIG="configs/pybullet/envs/official/domains/${env}.yaml"
        PLANNER_OUTPUT_PATH="${output_path}/${exp_name}/${CKPT}/${env}"
        # LOAD_PATH="${PLANNER_OUTPUT_PATH}/policy_cem"
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
