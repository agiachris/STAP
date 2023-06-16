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

function train_agent {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --agent-config ${AGENT_CONFIG}"
    args="${args} --env-config ${ENV_CONFIG}"
    if [ ! -z "${EVAL_ENV_CONFIG}" ]; then
        args="${args} --eval-env-config ${EVAL_ENV_CONFIG}"
    fi
    if [ ! -z "${ENCODER_CHECKPOINT}" ]; then
        args="${args} --encoder-checkpoint ${ENCODER_CHECKPOINT}"
    fi
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path ${POLICY_OUTPUT_PATH}_debug"
        args="${args} --overwrite"
        args="${args} --num-pretrain-steps 10"
        args="${args} --num-train-steps 10"
        args="${args} --num-eval-episodes 10"
    else
        args="${args} --path ${POLICY_OUTPUT_PATH}"
        if [[ $RESUME -ne 0 ]]; then
            args="${args} --resume"
        fi
        args="${args} --eval-recording-path ${EVAL_RECORDING_PATH}"
    fi

    CMD="python scripts/train/train_agent.py ${args}"
    run_cmd
}

# Setup.
SBATCH_SLURM="scripts/train/train_juno.sh"
DEBUG=0
RESUME=0

output_path="models"
plots_path="plots"
env_path="configs/pybullet/envs/official/primitives"

# Pybullet experiments.
if [[ `hostname` == *stanford.edu ]] || [[ `hostname` == juno* ]]; then
    ENV_KWARGS="--gui 0"
fi

# Train skill library.
# Details: light=200K episodes, MSE loss for Q-networks.
exp_name="primitives_light_mse"
AGENT_CONFIG="configs/pybullet/agents/single_stage/sac_mse.yaml"
TRAINER_CONFIG="configs/pybullet/trainers/agent/agent_light.yaml"
POLICY_OUTPUT_PATH="${output_path}/${exp_name}"
EVAL_RECORDING_PATH="${plots_path}/${exp_name}"

ENV_CONFIG="${env_path}/light/pick.yaml"
train_agent

ENV_CONFIG="${env_path}/light/place.yaml"
train_agent

ENV_CONFIG="${env_path}/light/pull.yaml"
train_agent

ENV_CONFIG="${env_path}/light/push.yaml"
train_agent

# Train skill library.
# Details: light=200K episodes, Logistic Regression loss for Q-networks.
# exp_name="primitives_light_logreg"
# AGENT_CONFIG="configs/pybullet/agents/single_stage/sac_mse.yaml"
# TRAINER_CONFIG="configs/pybullet/trainers/agent/agent_light.yaml"
# POLICY_OUTPUT_PATH="${output_path}/${exp_name}"
# EVAL_RECORDING_PATH="${plots_path}/${exp_name}"

# ENV_CONFIG="${env_path}/light/pick.yaml"
# train_agent

# ENV_CONFIG="${env_path}/light/place.yaml"
# train_agent

# ENV_CONFIG="${env_path}/light/pull.yaml"
# train_agent

# ENV_CONFIG="${env_path}/light/push.yaml"
# train_agent