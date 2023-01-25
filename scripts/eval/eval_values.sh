#!/bin/bash

set -e

function run_cmd {
    echo ""
    echo "${CMD}"
    eval "${CMD}"
}

function eval_value {
    args=""
    args="${args} --dataset-config ${DATA_CONFIG}"
    args="${args} --dataset-checkpoints ${DATA_CHECKPOINTS}"

    if [ ! -z "${TRAINER_CHECKPOINT}" ]; then
        args="${args} --trainer-checkpoint ${TRAINER_CHECKPOINT}"
    fi
    if [ ! -z "${AGENT_CHECKPOINT}" ]; then
        args="${args} --agent-checkpoint ${AGENT_CHECKPOINT}"
    fi
    if [ ! -z "${NAME}" ]; then
        args="${args} --name ${NAME}"
    fi

    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path ${VALUE_OUTPUT_PATH}_debug"
        args="${args} --overwrite"
        args="${args} --num-eval-steps 100"
    else
        args="${args} --path ${VALUE_OUTPUT_PATH}"
        if [ ! -z "${NUM_EVAL_STEPS}" ]; then
            args="${args} --num-eval-steps ${NUM_EVAL_STEPS}"
        fi
    fi

    CMD="python scripts/eval/eval_values.py ${args}"
    run_cmd
}

function run_value {
    DATA_CHECKPOINTS=""
    for seed in "${SEEDS[@]}"; do
        data_path="${DATA_CHECKPOINT_PATH}/train_${SYMBOLIC_ACTION_TYPE}_${PRIMITIVE}_${seed}/train_data"
        DATA_CHECKPOINTS="${TRAIN_DATA_CHECKPOINTS} ${data_path}"
    done

    NAME="${AGENT_CHECKPOINT}"
    AGENT_CHECKPOINT="${AGENT_CHECKPOINT_PATH}/${AGENT_CHECKPOINT}/final_model.pt"
    eval_value
}


#### Setup.
DEBUG=0
output_path="plots"

#### Experiments.

### Pybullet.
exp_name="20230124/value"
VALUE_OUTPUT_PATH="${output_path}/${exp_name}"
NUM_EVAL_STEPS=100

## Launch primitive jobs.

## Data.
DATA_CONFIG="configs/pybullet/datasets/replay_buffer.yaml"
SYMBOLIC_ACTION_TYPE="valid"
SEEDS=($(seq 0 1))

## Critics trained with mean squared regression.

# Pick w/out collisions, unbalanced data.
DATA_CHECKPOINT_PATH="models/20230116/datasets"
PRIMITIVE="pick"
AGENT_CHECKPOINT_PATH="models/20230120/value"
AGENT_CHECKPOINT="pick_value_sched-cos_iter-2M_sac_ens_value"
run_value

# Place w/out collisions, unbalanced data.
DATA_CHECKPOINT_PATH="models/20230116/datasets"
PRIMITIVE="place"
AGENT_CHECKPOINT_PATH="models/20230120/value"
AGENT_CHECKPOINT="place_value_sched-cos_iter-5M_sac_ens_value"
run_value

# Pull w/out collisions, unbalanced data.
DATA_CHECKPOINT_PATH="models/20230119/datasets"
PRIMITIVE="pull"
AGENT_CHECKPOINT_PATH="models/20230120/value"
AGENT_CHECKPOINT="pull_value_sched-cos_iter-2M_sac_ens_value"
run_value

# Push w/out collisions, unbalanced data.
DATA_CHECKPOINT_PATH="models/20230119/datasets"
PRIMITIVE="push"
AGENT_CHECKPOINT_PATH="models/20230120/value"
AGENT_CHECKPOINTS="push_value_sched-cos_iter-2M_sac_ens_value"
run_value

# Logistics regression

## Critics trained with logistics regression.

## Data.
# SYMBOLIC_ACTION_TYPE="valid"
# SEEDS=($(seq 0 1))

# # Pick w/out collisions, balanced data (40% success min).
# DATA_CHECKPOINT_PATH="models/20230124/datasets"
# PRIMITIVE="pick"
# CRITIC_CHECKPOINT_PATH="models/20230124/value"
# CRITIC_CHECKPOINTS=("pick/final_model")
# run_policy

# # Place w/out collisions, balanced data (40% success min).
# DATA_CHECKPOINT_PATH="models/20230124/datasets"
# PRIMITIVE="place"
# CRITIC_CHECKPOINT_PATH="models/20230124/value"
# CRITIC_CHECKPOINTS=("place/final_model")
# run_policy

# # Pull w/out collisions, balanced data (40% success min).
# DATA_CHECKPOINT_PATH="models/20230124/datasets"
# PRIMITIVE="pull"
# CRITIC_CHECKPOINT_PATH="models/20230124/value"
# CRITIC_CHECKPOINTS=("pull/final_model")
# run_policy

# # Push w/out collisions, balanced data (40% success min).
# DATA_CHECKPOINT_PATH="models/20230124/datasets"
# PRIMITIVE="push"
# CRITIC_CHECKPOINT_PATH="models/20230124/value"
# CRITIC_CHECKPOINTS=("push/final_model")
# run_policy
