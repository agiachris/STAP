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
        # args="${args} --eval-recording-path ${EVAL_RECORDING_PATH}_debug"
        args="${args} --overwrite"
        args="${args} --num-pretrain-steps 10"
        args="${args} --num-train-steps 100"
        args="${args} --num-eval-episodes 10"
    else
        args="${args} --path ${POLICY_OUTPUT_PATH}"
        args="${args} --eval-recording-path ${EVAL_RECORDING_PATH}"
    fi

    CMD="python scripts/train/train_policy.py ${args}"
    run_cmd
}

# Setup.
DEBUG=0
output_path="models"
plots_path="plots"

# Experiments.

# Pybox2d.
# exp_name="20220806/pybox2d"
# TRAINER_CONFIG="configs/pybox2d/trainers/agent.yaml"
# AGENT_CONFIG="configs/pybox2d/agents/sac.yaml"
# POLICY_OUTPUT_PATH="${output_path}/${exp_name}"
# EVAL_RECORDING_PATH="${plots_path}/${exp_name}"
#
# ENV_CONFIG="configs/pybox2d/envs/placeright.yaml"
# train_policy

# ENV_CONFIG="configs/pybox2d/envs/pushleft.yaml"
# train_policy

# Pybullet.
exp_name="20220904/examples_curriculum"
TRAINER_CONFIG="configs/pybullet/trainers/agent.yaml"
AGENT_CONFIG="configs/pybullet/agents/sac.yaml"
POLICY_OUTPUT_PATH="${output_path}/${exp_name}"
EVAL_RECORDING_PATH="${plots_path}/${exp_name}"
ENV_KWARGS="--num-env-processes 4 --num-eval-env-processes 2"
ENV_KWARGS="${ENV_KWARGS} --use-curriculum 1"
if [[ `hostname` == "sc.stanford.edu" ]]; then
    ENV_KWARGS="${ENV_KWARGS} --gui 0"
fi

ENV_CONFIG="configs/pybullet/envs/examples/primitives/pick.yaml"
EVAL_ENV_CONFIG="configs/pybullet/envs/examples/primitives/pick_eval.yaml"
train_policy

ENV_CONFIG="configs/pybullet/envs/examples/primitives/place.yaml"
EVAL_ENV_CONFIG="configs/pybullet/envs/examples/primitives/place_eval.yaml"
train_policy

ENV_CONFIG="configs/pybullet/envs/examples/primitives/pull.yaml"
EVAL_ENV_CONFIG="configs/pybullet/envs/examples/primitives/pull_eval.yaml"
train_policy
