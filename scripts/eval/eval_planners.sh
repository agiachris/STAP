#!/bin/bash

set -e

function run_cmd {
    echo ""
    echo "${CMD}"
    if [[ `hostname` == "sc.stanford.edu" ]]; then
        sbatch scripts/eval/eval_planners_juno.sh "${CMD}"
    else
        ${CMD}
    fi
}

function eval_planner {
    args=""
    args="${args} --planner-config ${PLANNER_CONFIG}"
    args="${args} --env-config ${ENV_CONFIG}"
    if [ ! -z "${POLICY_CHECKPOINTS}" ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS}"
    fi
    if [ ! -z "${SCOD_CHECKPOINTS}" ]; then
        args="${args} --scod-checkpoints ${SCOD_CHECKPOINTS}"
    fi
    if [ ! -z "${DYNAMICS_CHECKPOINT}" ]; then
        args="${args} --dynamics-checkpoint ${DYNAMICS_CHECKPOINT}"
    fi
    args="${args} --seed 0"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --num-eval 1"
        args="${args} --path ${SAVE_DIR}/plots_debug"
        args="${args} --verbose 1"
    else
        args="${args} --num-eval 100"
        args="${args} --path ${SAVE_DIR}/plots"
        args="${args} --verbose 0"
    fi
    CMD="python scripts/eval/eval_planners.py ${args}"
    run_cmd
}

function run_planners {
    for planner in "${PLANNERS[@]}"; do
        PLANNER_CONFIG="${CONFIG_PATH}/planners/${planner}.yaml"

        for train_step in ${TRAIN_STEPS[@]}; do
            SAVE_DIR="${OUTPUT_PATH}/${EXP_NAME}/planners_${train_step}"

            POLICY_CHECKPOINTS=()
            for policy_env in "${POLICY_ENVS[@]}"; do
                POLICY_CHECKPOINTS+=("${OUTPUT_PATH}/${EXP_NAME}/${policy_env}/ckpt_model_${train_step}.pt")
            done
            POLICY_CHECKPOINTS="${POLICY_CHECKPOINTS[@]}"
        
            if [[ "${planner}" == *scod_value* ]]; then
                SCOD_CHECKPOINTS=()
                for policy_env in "${POLICY_ENVS[@]}"; do
                    SCOD_CHECKPOINTS+=("${OUTPUT_PATH}/${EXP_NAME}/${policy_env}/scod_${train_step}/final_scod.pt")
                done
                SCOD_CHECKPOINTS="${SCOD_CHECKPOINTS[@]}"
            else
                SCOD_CHECKPOINTS=""
            fi

            if [[ "${planner}" == *_oracle_*dynamics ]] || [[ "${planner}" == "random" ]]; then
                DYNAMICS_CHECKPOINT=""
            else
                DYNAMICS_CHECKPOINT="${OUTPUT_PATH}/${EXP_NAME}/dynamics_${train_step}/dynamics/best_model.pt"
            fi

            eval_planner
        done
    done
}

function visualize_results {
    args=""
    args="${args} --path ${SAVE_DIR}/plots"
    args="${args} --methods ${PLANNERS[@]}"
    CMD="python scripts/visualize/visualize_planners.py ${args}"
    run_cmd
}

# Setup.
DEBUG=0
OUTPUT_PATH="${OUTPUTS}/temporal_policies/models"
CONFIG_PATH="configs/pybox2d"

# Policies / experiments.
EXP_NAME="20220727/decoupled_state"
ENV_CONFIG="${CONFIG_PATH}/envs/placeright_pushleft.yaml"
POLICY_ENVS=("placeright" "pushleft")
TRAIN_STEPS=(50000 100000)

# Evaluate planners.
PLANNERS=(
# Q-value / Latent dynamics.
    # "policy_cem"
    # "random_cem"
    # "policy_shooting"
    # "random_shooting"
# Oracle value / Latent dynamics.
    # "policy_cem_oracle_value"
    # "random_cem_oracle_value"
    # "policy_shooting_oracle_value"
    # "random_shooting_oracle_value"
# Q-value / Oracle dynamics.
    "policy_cem_oracle_dynamics"
    "random_cem_oracle_dynamics"
    "policy_shooting_oracle_dynamics"
    "random_shooting_oracle_dynamics"
# Oracle value / Oracle dynamics.
    # "policy_cem_oracle_value_dynamics"
    # "random_cem_oracle_value_dynamics"
    # "policy_shooting_oracle_value_dynamics"
    # "random_shooting_oracle_value_dynamics"
# SCOD value / Oracle dynamics
    "policy_cem_var_scod_value_oracle_dynamics"
    "policy_shooting_var_scod_value_oracle_dynamics"
    "policy_cem_cvar_scod_value_oracle_dynamics"
    "policy_shooting_cvar_scod_value_oracle_dynamics"
# SCOD value / Latent dynamics
    # "policy_cem_scod_value"
    # "policy_shooting_scod_value"
# Random.
    "random"
)
run_planners

# Visualize results.
if [[ `hostname` == "sc.stanford.edu" ]]; then
    exit
fi

for train_step in $TRAIN_STEPS; do
    SAVE_DIR="${OUTPUT_PATH}/${EXP_NAME}/planners_${train_step}"
    visualize_results
done
