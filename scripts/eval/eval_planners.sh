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
    args="${args} --path plots/pybox2d"
    args="${args} --methods ${PLANNERS[@]}"
    CMD="python scripts/visualize/visualize_planners.py ${args}"
    run_cmd
}

# Setup.

DEBUG=1

declare -a PLANNERS=(
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
    # "policy_cem_oracle_dynamics"
    # "random_cem_oracle_dynamics"
    # "policy_shooting_oracle_dynamics"
    # "random_shooting_oracle_dynamics"
# Oracle value / Oracle dynamics.
    # "policy_cem_oracle_value_dynamics"
    # "random_cem_oracle_value_dynamics"
    # "policy_shooting_oracle_value_dynamics"
    # "random_shooting_oracle_value_dynamics"
# Random.
    "random"
)

# Evaluate planners.

for EXP_NAME in "20220426/unified 20220426/decoupled"; do
for planner in "${PLANNERS[@]}"; do
    ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft_img.yaml
    POLICY_CHECKPOINTS="models/${EXP_NAME}/placeright_img/final_model.pt models/${EXP_NAME}/pushleft_img/final_model.pt"
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

declare -a PLANNERS=(
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

visualize_results
