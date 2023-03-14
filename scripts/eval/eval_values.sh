#!/bin/bash

set -e

function run_cmd {
    echo ""
    echo "${CMD}"
    if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == juno* ]]; then
        sbatch "${SBATCH_SLURM}" "${CMD}"
    elif [[ `hostname` == "${GCP_LOGIN}" ]]; then
        sbatch scripts/train/train_gcp.sh "${CMD}"
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

output_path="plots"
exp_name="20230309/value_logits"
VALUE_OUTPUT_PATH="${output_path}/${exp_name}"
NUM_EVAL_STEPS=1000

# Critics trained with logistics regression, balanced data (40%).
DATA_CONFIG="configs/pybullet/datasets/replay_buffer.yaml"
AGENT_CONFIG="configs/pybullet/agents/multi_stage/value/sac_ens_value_logistics_logit.yaml"
SEEDS=("0")
declare -A AGENT_CHECKPOINT_PATHS=(
    ["pick"]="models/20230309/value"
    ["place"]="models/20230309/value"
    ["pull"]="models/20230309/value"
    ["push"]="models/20230309/value"
)
for PRIMITIVE in "${!AGENT_CHECKPOINT_PATHS[@]}"; do
    AGENT_CHECKPOINT_PATH="${AGENT_CHECKPOINT_PATHS[${PRIMITIVE}]}"
    
    AGENT_CHECKPOINT="${PRIMITIVE}"
    SYMBOLIC_ACTION_TYPE="valid"
    DATA_CHECKPOINT_PATH="models/20230309/datasets"
    TAG="ind"
    run_value
    
    AGENT_CHECKPOINT="${PRIMITIVE}"
    SYMBOLIC_ACTION_TYPE="invalid"
    DATA_CHECKPOINT_PATH="models/20230309/datasets"
    TAG="ood"
    run_value
done