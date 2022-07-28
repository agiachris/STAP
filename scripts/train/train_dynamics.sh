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
    args="${args} --path models/${EXP_NAME}"
    args="${args} --seed 0"
    # args="${args} --overwrite"

    CMD="python scripts/train/train_dynamics.py ${args}"
    run_cmd
}

# for train_step in 50000 100000 150000 200000; do
#     EXP_NAME="20220428/decoupled_state"
#     TRAINER_CONFIG="configs/pybox2d/trainers/dynamics.yaml"
#     DYNAMICS_CONFIG="configs/pybox2d/dynamics/decoupled.yaml"
#     POLICY_CHECKPOINTS=(
#         "models/${EXP_NAME}/placeright/ckpt_model_${train_step}.pt"
#         "models/${EXP_NAME}/pushleft/ckpt_model_${train_step}.pt"
#     )
#     EXP_NAME="${EXP_NAME}/dynamics_${train_step}"
#     train_dynamics
# done

# for train_step in 50000 100000 150000 200000; do
#     EXP_NAME="20220428/decoupled_img"
#     TRAINER_CONFIG="configs/pybox2d/trainers/dynamics.yaml"
#     DYNAMICS_CONFIG="configs/pybox2d/dynamics/shared.yaml"
#     POLICY_CHECKPOINTS=(
#         "models/${EXP_NAME}/placeright_img/ckpt_model_${train_step}.pt"
#         "models/${EXP_NAME}/pushleft_img/ckpt_model_${train_step}.pt"
#     )
#     EXP_NAME="${EXP_NAME}/dynamics_${train_step}"
#     train_dynamics
# done

# EXP_NAME="20220721/pybox2d"
# TRAINER_CONFIG="configs/pybox2d/trainers/dynamics.yaml"
# DYNAMICS_CONFIG="configs/pybox2d/dynamics/shared.yaml"
# POLICY_CHECKPOINTS=(
#     "models/${EXP_NAME}/placeright/best_model.pt"
#     "models/${EXP_NAME}/pushleft/best_model.pt"
# )
# train_dynamics

EXP_NAME="20220726/workspace"
TRAINER_CONFIG="configs/pybullet/trainers/dynamics.yaml"
DYNAMICS_CONFIG="configs/pybullet/dynamics/table_env.yaml"
POLICY_CHECKPOINTS=(
    "models/${EXP_NAME}/pick/final_model.pt"
    "models/${EXP_NAME}/place/final_model.pt"
    "models/${EXP_NAME}/pull/final_model.pt"
)
train_dynamics
