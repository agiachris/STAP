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

function train_policy {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --agent-config ${AGENT_CONFIG}"
    args="${args} --env-config ${ENV_CONFIG}"
    args="${args} --critic-checkpoint ${CRITIC_CHECKPOINT}"
    if [ ! -z "${ENCODER_CHECKPOINT}" ]; then
        args="${args} --encoder-checkpoint ${ENCODER_CHECKPOINT}"
    fi
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"    
    args="${args} --train-data-checkpoints ${TRAIN_DATA_CHECKPOINTS}"
    args="${args} --eval-data-checkpoints ${EVAL_DATA_CHECKPOINTS}"
    if [ ! -z "${NAME}" ]; then
        args="${args} --name ${NAME}"
    fi
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path ${POLICY_OUTPUT_PATH}_debug"
        args="${args} --overwrite"
        args="${args} --num-train-steps 10"
        args="${args} --num-eval-episodes 10"
    else
        args="${args} --path ${POLICY_OUTPUT_PATH}"
        args="${args} --eval-recording-path ${EVAL_RECORDING_PATH}"
    fi

    CMD="python scripts/train/train_agent.py ${args}"
    run_cmd
}

function run_policy {
    ENV_CONFIG="configs/pybullet/envs/official/primitives/heavy/${PRIMITIVE}_eval.yaml"

    TRAIN_DATA_CHECKPOINTS=""
    for seed in "${TRAIN_SEEDS[@]}"; do
        data_path="${DATA_CHECKPOINT_PATH}/train_${SYMBOLIC_ACTION_TYPE}_${PRIMITIVE}_${seed}/train_data"
        TRAIN_DATA_CHECKPOINTS="${TRAIN_DATA_CHECKPOINTS} ${data_path}"
    done

    EVAL_DATA_CHECKPOINTS=""
    for seed in "${VALIDATION_SEEDS[@]}"; do
        data_path="${DATA_CHECKPOINT_PATH}/validation_${SYMBOLIC_ACTION_TYPE}_${PRIMITIVE}_${seed}/train_data"
        EVAL_DATA_CHECKPOINTS="${EVAL_DATA_CHECKPOINTS} ${data_path}"
    done    

    CRITIC_CHECKPOINT="${CRITIC_CHECKPOINT_PATH}/${PRIMITIVE}/${CHECKPOINT}.pt"

    train_policy
}

# Setup.
SBATCH_SLURM="scripts/train/train_juno.sh"
DEBUG=0

input_path="models"
output_path="models"
plots_path="plots"

# Pybullet experiments.
if [[ `hostname` == *stanford.edu ]] || [[ `hostname` == juno* ]]; then
    ENV_KWARGS="--gui 0"
fi

# Train policy library.
exp_name="policies_irl"
POLICY_OUTPUT_PATH="${output_path}/${exp_name}"
EVAL_RECORDING_PATH="${plots_path}/${exp_name}"

SYMBOLIC_ACTION_TYPE="valid"
TRAIN_SEEDS=($(seq 0 15))
VALIDATION_SEEDS=($(seq 16 19))
DATA_CHECKPOINT_PATH="${input_path}/datasets"

AGENT_CONFIG="configs/pybullet/agents/multi_stage/policy/sac_policy.yaml"
TRAINER_CONFIG="configs/pybullet/trainers/policy/policy.yaml"

CRITIC_CHECKPOINT_PATH="${input_path}/value_fns_irl"
CHECKPOINT="final_model"

# Details: 1M episodes, Logistic Regression loss for Q-networks, ensemble of 8 Q-networks.
PRIMITIVE="pick"
run_policy

# Details: 1M episodes, MSE loss for Q-networks, ensemble of 8 Q-networks.
PRIMITIVE="place"
run_policy

# Details: 1M episodes, MSE loss for Q-networks, ensemble of 8 Q-networks.
PRIMITIVE="pull"
run_policy

# Details: 1M episodes, MSE loss for Q-networks, ensemble of 8 Q-networks.
PRIMITIVE="push"
run_policy