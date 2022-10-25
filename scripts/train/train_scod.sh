#!/bin/bash

set -e

GCP_LOGIN="juno-login-lclbjqwy-001"

function run_cmd {
    echo ""
    echo "${CMD}"
    if [[ `hostname` == "sc.stanford.edu" ]]; then
        sbatch scripts/train/train_juno.sh "${CMD}"
    elif [[ `hostname` == "${GCP_LOGIN}" ]]; then
        sbatch scripts/train/train_gcp.sh "${CMD}"
    else
        ${CMD}
    fi
}

function train_scod {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --scod-config ${SCOD_CONFIG}"
    args="${args} --model-checkpoint ${MODEL_CHECKPOINT}"
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"

    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path ${SCOD_OUTPUT_PATH}_debug"
        args="${args} --overwrite"
    else
        args="${args} --path ${SCOD_OUTPUT_PATH}"
    fi

    CMD="python scripts/train/train_scod.py ${args}"
    run_cmd
}

# Setup.
DEBUG=0
output_path="models"

# Experiments.

exp_name="20220914/official"
TRAINER_CONFIG="configs/pybullet/trainers/scod.yaml"
scod_configs=(
    "scod"
    # "scod_freeze"
    # "scod_srft"
    # "scod_srft_freeze"
)
policy_envs=("pick" "place" "pull" "push")
checkpoints=(
    # "final_model"
    # "best_model"
    # "select_model"
    "ckpt_model_50000"
    "ckpt_model_100000"
    "ckpt_model_150000"
    "ckpt_model_200000"
    # "selectscod_model"
    # "selectscodfreeze_model"
    # "selectscodsrft_model"
    # "selectscodsrftfreeze_model"
)

if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]]; then
    ENV_KWARGS="--gui 0"
fi

for ckpt in "${checkpoints[@]}"; do
    for policy_env in "${policy_envs[@]}"; do
        MODEL_CHECKPOINT="${output_path}/${exp_name}/${policy_env}/${ckpt}.pt"
        for scod_config in "${scod_configs[@]}"; do
            SCOD_CONFIG="configs/pybullet/scod/${scod_config}.yaml"
            SCOD_OUTPUT_PATH="${output_path}/${exp_name}/${ckpt}/${policy_env}/${scod_config}"
            train_scod
        done
    done
done

