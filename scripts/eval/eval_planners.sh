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
    ARGS="${ARGS} --seed 0"
    ARGS="${ARGS} --num-eval 100"
    ARGS="${ARGS} --path plots/pybox2d"
    CMD="python scripts/eval/eval_planners.py ${ARGS}"
    echo ""
    echo "${CMD}"
    ${CMD}
}

function visualize_results {
    ARGS=""
    ARGS="${ARGS} --path plots/pybox2d"
    METHODS=""
    METHODS="${METHODS} policy_cem"
    METHODS="${METHODS} random_cem"
    METHODS="${METHODS} policy_shooting"
    METHODS="${METHODS} random_shooting"
    METHODS="${METHODS} policy_cem_oracle_dynamics"
    METHODS="${METHODS} random_cem_oracle_dynamics"
    METHODS="${METHODS} policy_shooting_oracle_dynamics"
    METHODS="${METHODS} random_shooting_oracle_dynamics"
    METHODS="${METHODS} policy_cem_oracle_value_dynamics"
    METHODS="${METHODS} random_cem_oracle_value_dynamics"
    METHODS="${METHODS} policy_shooting_oracle_value_dynamics"
    METHODS="${METHODS} random_shooting_oracle_value_dynamics"
    METHODS="${METHODS} random"
    ARGS="${ARGS} --methods ${METHODS}"
    CMD="python scripts/visualize/visualize_planners.py ${ARGS}"
    echo ""
    echo "${CMD}"
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

PLANNER_CONFIG=configs/pybox2d/planners/random_cem.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
DYNAMICS_CHECKPOINT="models/placeright_pushleft/final_model.pt"
eval_planner

PLANNER_CONFIG=configs/pybox2d/planners/random_shooting.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
DYNAMICS_CHECKPOINT="models/placeright_pushleft/final_model.pt"
eval_planner

# Oracle value.

# PLANNER_CONFIG=configs/pybox2d/planners/policy_cem_oracle_value.yaml
# ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
# POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
# DYNAMICS_CHECKPOINT="models/placeright_pushleft/final_model.pt"
# eval_planner
#
# PLANNER_CONFIG=configs/pybox2d/planners/policy_shooting_oracle_value.yaml
# ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
# POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
# DYNAMICS_CHECKPOINT="models/placeright_pushleft/final_model.pt"
# eval_planner
#
# PLANNER_CONFIG=configs/pybox2d/planners/random_cem_oracle_value.yaml
# ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
# POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
# DYNAMICS_CHECKPOINT="models/placeright_pushleft/final_model.pt"
# eval_planner
#
# PLANNER_CONFIG=configs/pybox2d/planners/random_shooting_oracle_value.yaml
# ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
# POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
# DYNAMICS_CHECKPOINT="models/placeright_pushleft/final_model.pt"
# eval_planner

# Oracle dynamics.

PLANNER_CONFIG=configs/pybox2d/planners/policy_cem_oracle_dynamics.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
DYNAMICS_CHECKPOINT=""
eval_planner

PLANNER_CONFIG=configs/pybox2d/planners/policy_shooting_oracle_dynamics.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
DYNAMICS_CHECKPOINT=""
eval_planner

PLANNER_CONFIG=configs/pybox2d/planners/random_cem_oracle_dynamics.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
DYNAMICS_CHECKPOINT=""
eval_planner

PLANNER_CONFIG=configs/pybox2d/planners/random_shooting_oracle_dynamics.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
DYNAMICS_CHECKPOINT=""
eval_planner

# Oracle value and dynamics.

PLANNER_CONFIG=configs/pybox2d/planners/policy_cem_oracle_value_dynamics.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
DYNAMICS_CHECKPOINT=""
eval_planner

PLANNER_CONFIG=configs/pybox2d/planners/policy_shooting_oracle_value_dynamics.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
DYNAMICS_CHECKPOINT=""
eval_planner

PLANNER_CONFIG=configs/pybox2d/planners/random_cem_oracle_value_dynamics.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
DYNAMICS_CHECKPOINT=""
eval_planner

PLANNER_CONFIG=configs/pybox2d/planners/random_shooting_oracle_value_dynamics.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
DYNAMICS_CHECKPOINT=""
eval_planner

# Random.

PLANNER_CONFIG=configs/pybox2d/planners/random.yaml
ENV_CONFIG=configs/pybox2d/envs/placeright_pushleft.yaml
POLICY_CHECKPOINTS="models/placeright/final_model.pt models/pushleft/final_model.pt"
DYNAMICS_CHECKPOINT="models/placeright_pushleft/final_model.pt"
eval_planner

visualize_results
