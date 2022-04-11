#!/bin/bash

set -e

function eval_planner {
    ARGS=""
    ARGS="${ARGS} --planner-config ${PLANNER_CONFIG}"
    ARGS="${ARGS} --env-config ${ENV_CONFIG}"
    if [ ! -z "${POLICY_CHECKPOINTS}" ]; then
        ARGS="${ARGS} --policy-checkpoints ${POLICY_CHECKPOINTS}"
    fi
    if [ ! -z "${DYNAMICS_CHECKPOINT}" ]; then
        ARGS="${ARGS} --dynamics-checkpoint ${DYNAMICS_CHECKPOINT}"
    fi
    CMD="python scripts/eval/eval_planners.py ${ARGS}"
    echo ${CMD}
    ${CMD}
}

PLANNER_CONFIG=configs/pybox2d/planners/policy_cem.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
DYNAMICS_CHECKPOINT="models/placeright_pushleft/final_model.pt"
eval_planner

PLANNER_CONFIG=configs/pybox2d/planners/policy_shooting.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
DYNAMICS_CHECKPOINT="models/placeright_pushleft/final_model.pt"
eval_planner

PLANNER_CONFIG=configs/pybox2d/planners/oracle_policy_shooting.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
DYNAMICS_CHECKPOINT=""
eval_planner

PLANNER_CONFIG=configs/pybox2d/planners/random_shooting.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
DYNAMICS_CHECKPOINT="models/placeright_pushleft/final_model.pt"
eval_planner

PLANNER_CONFIG=configs/pybox2d/planners/oracle_random_shooting.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
DYNAMICS_CHECKPOINT=""
eval_planner

PLANNER_CONFIG=configs/pybox2d/planners/oracle_shooting.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS=""
DYNAMICS_CHECKPOINT=""
eval_planner

PLANNER_CONFIG=configs/pybox2d/planners/random.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS=""
DYNAMICS_CHECKPOINT=""
eval_planner
