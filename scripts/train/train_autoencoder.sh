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

function train_autoencoder {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --encoder-config ${ENCODER_CONFIG}"
    if [ ${#POLICY_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS[@]}"
    fi
    args="${args} --path models/${EXP_NAME}"
    args="${args} --seed 0"
    # args="${args} --overwrite"

    CMD="python scripts/train/train_autoencoder.py ${args}"
    run_cmd
}

EXP_NAME="20220513/vae_conv128_inv"
TRAINER_CONFIG="configs/pybox2d/trainers/autoencoder.yaml"
ENCODER_CONFIG="configs/pybox2d/encoders/vae.yaml"
POLICY_CHECKPOINTS=(
    "models/20220428/decoupled_img/placeright_img/final_model.pt"
    "models/20220428/decoupled_img/pushleft_img/final_model.pt"
)
train_autoencoder
