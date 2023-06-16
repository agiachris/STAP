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

function train_dynamics {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --dynamics-config ${DYNAMICS_CONFIG}"
    if [ ${#POLICY_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS[@]}"
    fi
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"

    if [ ! -z "${NAME}" ]; then
        args="${args} --name ${NAME}"
    fi
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path ${DYNAMICS_OUTPUT_PATH}_debug"
        args="${args} --overwrite"
        args="${args} --num-train-steps 10000"
    else
        args="${args} --path ${DYNAMICS_OUTPUT_PATH}"
    fi

    CMD="python scripts/train/train_dynamics.py ${args}"
    run_cmd
}

function run_dynamics {
    NAME=""
    POLICY_CHECKPOINTS=()
    for primitive in "${PRIMITIVES[@]}"; do
        POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${primitive}/${POLICY_CHECKPOINT}.pt")

        if [ -z "${NAME}" ]; then
            NAME="${primitive}"
        else
            NAME="${NAME}_${primitive}"
        fi
    done
    NAME="${NAME}_dynamics"

    train_dynamics
}

# Setup.
SBATCH_SLURM="scripts/train/train_juno.sh"
DEBUG=0

input_path="models"
output_path="models"

# Pybullet experiments.
if [[ `hostname` == *stanford.edu ]] || [[ `hostname` == juno* ]]; then
    ENV_KWARGS="--gui 0"
fi

# Train dynamics.
exp_name="dynamics"
DYNAMICS_OUTPUT_PATH="${output_path}/${exp_name}"
DYNAMICS_CONFIG="configs/pybullet/dynamics/table_env.yaml"
TRAINER_CONFIG="configs/pybullet/trainers/dynamics/dynamics_iter-0.75M.yaml"

PRIMITIVES=(
    "pick"
    "place"
    "pull"
    "push"
)
POLICY_INPUT_PATH="${input_path}/primitives_light_mse"
POLICY_CHECKPOINT="final_model"
run_dynamics