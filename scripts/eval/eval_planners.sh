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
    if [ ! -z "${DYNAMICS_CHECKPOINT}" ]; then
        args="${args} --dynamics-checkpoint ${DYNAMICS_CHECKPOINT}"
    fi
    args="${args} --seed 0"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --num-eval 1"
        args="${args} --path plots/${EXP_NAME}_debug"
        args="${args} --verbose 1"
    else
        args="${args} --num-eval 100"
        args="${args} --path plots/${EXP_NAME}"
        args="${args} --verbose 0"
    fi
    CMD="python scripts/eval/eval_planners.py ${args}"
    run_cmd
}

function visualize_results {
    args=""
    args="${args} --path plots/${EXP_NAME}"
    args="${args} --methods ${PLANNERS[@]}"
    CMD="python scripts/visualize/visualize_planners.py ${args}"
    run_cmd
}

# Setup.

DEBUG=0

PLANNERS=(
# Q-value / Latent dynamics.
    "policy_cem"
    "random_cem"
    "policy_shooting"
    "random_shooting"
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
    "policy_cem_oracle_value_dynamics"
    "random_cem_oracle_value_dynamics"
    "policy_shooting_oracle_value_dynamics"
    "random_shooting_oracle_value_dynamics"
# Random.
    "random"
)

# Evaluate planners.

planner_env="placeright_pushleft_img"
policy_envs=(
    "placeright_img"
    "pushleft_img"
)
experiments=(
    "20220426/unified"
    "20220426/decoupled"
)

for EXP_NAME in "${experiments[@]}"; do
    for planner in "${PLANNERS[@]}"; do
        ENV_CONFIG=configs/pybox2d/envs/${planner_env}.yaml
        POLICY_CHECKPOINTS=()
        for policy_env in "${policy_envs[@]}"; do
            POLICY_CHECKPOINTS+=("models/${EXP_NAME}/${policy_env}/final_model.pt")
        done
        POLICY_CHECKPOINTS="${POLICY_CHECKPOINTS[@]}"
        if [[ "${planner}" == *_oracle_*dynamics ]] || [[ "${planner}" == "random" ]]; then
            DYNAMICS_CHECKPOINT=""
        else
            DYNAMICS_CHECKPOINT="models/${EXP_NAME}/dynamics/final_model.pt"
        fi
        PLANNER_CONFIG="configs/pybox2d/planners/${planner}.yaml"

        eval_planner
    done
done

if [[ `hostname` == "sc.stanford.edu" ]]; then
    exit
fi

# Visualize results.

PLANNERS=(
# Q-value / Latent dynamics.
    "policy_cem"
    "random_cem"
    "policy_shooting"
    "random_shooting"
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
    "policy_cem_oracle_value_dynamics"
    "random_cem_oracle_value_dynamics"
    "policy_shooting_oracle_value_dynamics"
    "random_shooting_oracle_value_dynamics"
# Random.
    "random"
)

for EXP_NAME in "${experiments[@]}"; do
    visualize_results
done
