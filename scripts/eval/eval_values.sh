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
    if [ ! -z "${AGENT_CONFIG}" ]; then
        args="${args} --agent-config ${AGENT_CONFIG}"
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
        args="${args} --overwrite"
    fi

    CMD="python scripts/eval/eval_values.py ${args}"
    run_cmd
}

function run_value {
    DATA_CHECKPOINTS=""
    for seed in "${SEEDS[@]}"; do
        data_path="${DATA_CHECKPOINT_PATH}/train_${SYMBOLIC_ACTION_TYPE}_${PRIMITIVE}_${seed}/train_data"
        DATA_CHECKPOINTS="${DATA_CHECKPOINTS} ${data_path}"
    done

    NAME="${AGENT_CHECKPOINT}"
    if [ ! -z "${TAG}" ]; then
        NAME="${NAME}_${TAG}"
    fi
    AGENT_CHECKPOINT="${AGENT_CHECKPOINT_PATH}/${AGENT_CHECKPOINT}/final_model.pt"
    eval_value
}

# Setup.
SBATCH_SLURM="scripts/train/train_juno.sh"
DEBUG=0

input_path="models"
output_path="plots"
exp_name="value_fns_irl"

VALUE_INPUT_PATH="${input_path}/value_fns_irl"
DATA_CHECKPOINT_PATH="${input_path}/datasets"
VALUE_OUTPUT_PATH="${output_path}/${exp_name}"
DATA_CONFIG="configs/pybullet/datasets/replay_buffer.yaml"

NUM_EVAL_STEPS=1000

SEEDS=("0")
PRIMITIVES=(
    "pick"
    "place"
    "pull"
    "push"
)
for PRIMITIVE in "${PRIMITIVES[@]}"; do
    AGENT_CHECKPOINT_PATH="${VALUE_INPUT_PATH}"
    
    AGENT_CHECKPOINT="${PRIMITIVE}"
    SYMBOLIC_ACTION_TYPE="valid"
    TAG="ind"
    run_value
    
    AGENT_CHECKPOINT="${PRIMITIVE}"
    SYMBOLIC_ACTION_TYPE="invalid"
    TAG="ood"
    run_value
done