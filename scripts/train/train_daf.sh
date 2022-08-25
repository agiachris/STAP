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

function train_daf {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --dynamics-config ${DYNAMICS_CONFIG}"
    args="${args} --agent-config ${AGENT_CONFIG}"
    args="${args} --env-config ${ENV_CONFIG}"
    args="${args} --planner-config ${PLANNER_CONFIG}"
    if [ ! -z "${POLICY_CHECKPOINTS}" ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS}"
    fi
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path ${OUTPUT_PATH}_debug"
        args="${args} --eval-recording-path ${EVAL_RECORDING_PATH}_debug"
        args="${args} --overwrite"
        args="${args} --num-pretrain-steps 10"
        args="${args} --num-train-steps 10"
        args="${args} --num-eval-steps 10"
    else
        args="${args} --path ${OUTPUT_PATH}"
        args="${args} --eval-recording-path ${EVAL_RECORDING_PATH}"
    fi

    CMD="python scripts/train/train_daf.py ${args}"
    run_cmd
}

# Setup.
DEBUG=0
output_path="models"
plots_path="plots"

# Experiments.

exp_name="20220816/workspace_daf"
TRAINER_CONFIG="configs/pybullet/trainers/daf.yaml"
DYNAMICS_CONFIG="configs/pybullet/dynamics/table_env.yaml"
AGENT_CONFIG="configs/pybullet/agents/sac.yaml"
ENV_CONFIG="configs/pybullet/envs/workspace.yaml"
PLANNER_CONFIG="configs/pybullet/planners/policy_shooting.yaml"
OUTPUT_PATH="${output_path}/${exp_name}"
EVAL_RECORDING_PATH="${plots_path}/${exp_name}"
if [[ `hostname` == "sc.stanford.edu" ]]; then
    ENV_KWARGS="--gui 0"
fi

train_daf
