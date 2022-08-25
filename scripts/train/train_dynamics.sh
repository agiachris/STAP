#!/bin/bash

set -e

function run_cmd {
    echo ""
    echo "${CMD}"
    if [[ `hostname` == "sc.stanford.edu" ]]; then
        sbatch scripts/train/train_juno.sh "${CMD}"
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

    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path ${DYNAMICS_OUTPUT_PATH}_debug"
        args="${args} --overwrite"
    else
        args="${args} --path ${DYNAMICS_OUTPUT_PATH}"
    fi

    CMD="python scripts/train/train_dynamics.py ${args}"
    run_cmd
}

# Setup.
DEBUG=0
output_path="models"

# Experiments.

# exp_name="20220721/pybox2d"
# TRAINER_CONFIG="configs/pybox2d/trainers/dynamics.yaml"
# DYNAMICS_CONFIG="configs/pybox2d/dynamics/shared.yaml"
# policy_envs=("placeright" "pushleft")
# checkpoints=(
#     "final_model"
#     "best_model"
#     "ckpt_model_50000"
#     "ckpt_model_100000"
# )

exp_name="20220801/workspace"
TRAINER_CONFIG="configs/pybullet/trainers/dynamics.yaml"
DYNAMICS_CONFIG="configs/pybullet/dynamics/table_env.yaml"
policy_envs=("pick" "place" "pull")
checkpoints=(
    "final_model"
    "best_model"
    "ckpt_model_50000"
    "ckpt_model_100000"
)

for ckpt in "${checkpoints[@]}"; do
    POLICY_CHECKPOINTS=()
    for policy_env in "${policy_envs[@]}"; do
        POLICY_CHECKPOINTS+=("${output_path}/${exp_name}/${policy_env}/${ckpt}.pt")
    done
    POLICY_CHECKPOINTS="${POLICY_CHECKPOINTS[@]}"

    DYNAMICS_OUTPUT_PATH="${output_path}/${exp_name}/${ckpt}"
    train_dynamics
done
