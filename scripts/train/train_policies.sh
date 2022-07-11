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

function train_policy {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --agent-config ${AGENT_CONFIG}"
    args="${args} --env-config ${ENV_CONFIG}"
    # args="${args} --encoder-checkpoint ${ENCODER_CHECKPOINT}"
    args="${args} --path models/${EXP_NAME}"
    args="${args} --seed 0"
    # args="${args} --overwrite"

    CMD="python scripts/train/train_policy.py ${args}"
    run_cmd
}

# EXP_NAME="20220428/decoupled_state_debug"
#
# TRAINER_CONFIG="configs/pybox2d/trainers/agent.yaml"
# AGENT_CONFIG="configs/pybox2d/agents/sac.yaml"
# ENV_CONFIG="configs/pybox2d/envs/placeright.yaml"
# train_policy
#
# TRAINER_CONFIG="configs/pybox2d/trainers/agent.yaml"
# AGENT_CONFIG="configs/pybox2d/agents/sac.yaml"
# ENV_CONFIG="configs/pybox2d/envs/pushleft.yaml"
# train_policy

# EXP_NAME="20220428/decoupled_img_debug"
#
# TRAINER_CONFIG="configs/pybox2d/trainers/agent.yaml"
# AGENT_CONFIG="configs/pybox2d/agents/sac_img.yaml"
# ENV_CONFIG="configs/pybox2d/envs/placeright_img.yaml"
# train_policy
#
# TRAINER_CONFIG="configs/pybox2d/trainers/agent.yaml"
# AGENT_CONFIG="configs/pybox2d/agents/sac_img.yaml"
# ENV_CONFIG="configs/pybox2d/envs/pushleft_img.yaml"
# train_policy

# EXP_NAME="20220510/vae"
#
# TRAINER_CONFIG="configs/pybox2d/trainers/agent.yaml"
# AGENT_CONFIG="configs/pybox2d/agents/sac_vae.yaml"
# ENV_CONFIG="configs/pybox2d/envs/placeright_img.yaml"
# ENCODER_CHECKPOINT="models/20220510/vae/encoder/final_model.pt"
# train_policy
#
# TRAINER_CONFIG="configs/pybox2d/trainers/agent.yaml"
# AGENT_CONFIG="configs/pybox2d/agents/sac.yaml"
# ENV_CONFIG="configs/pybox2d/envs/pushleft_img.yaml"
# train_policy

EXP_NAME="20220708/pick_hook"

TRAINER_CONFIG="configs/pybullet/trainers/agent.yaml"
AGENT_CONFIG="configs/pybullet/agents/sac.yaml"
ENV_CONFIG="configs/pybullet/envs/pick.yaml"
train_policy
