#!/bin/bash

set -e

# GCP_LOGIN="juno-login-lclbjqwy-001"
# GCP_LOGIN="gcp-login-yq0fvtuw-001"

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

    CMD="python scripts/train/train_policy.py ${args}"
    run_cmd
}

function run_policy {
    ENV_CONFIG="configs/pybullet/envs/t2m/official/primitives/primitives_rl/${PRIMITIVE}_eval.yaml"
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

    for critic_checkpoint in "${CRITIC_CHECKPOINTS}"; do
        CRITIC_CHECKPOINT="${CRITIC_CHECKPOINT_PATH}/${critic_checkpoint}.pt"
        NAME="${critic_checkpoint}"
        train_policy
    done
}

#### Setup.
SBATCH_SLURM="scripts/train/train_juno.sh"
DEBUG=0
output_path="models"
plots_path="plots"

#### Experiments.

### Pybullet.
exp_name="20230123/policy"
POLICY_OUTPUT_PATH="${output_path}/${exp_name}"
EVAL_RECORDING_PATH="${plots_path}/${exp_name}"

TRAINER_CONFIG="configs/pybullet/trainers/policy.yaml"
AGENT_CONFIG="configs/pybullet/agents/multi_stage/sac_policy.yaml"
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]] || [[ `hostname` == juno* ]]; then
    ENV_KWARGS="--gui 0"
fi

## Data.
SYMBOLIC_ACTION_TYPE="valid"
TRAIN_SEEDS=($(seq 0 7))
VALIDATION_SEEDS=($(seq 8 9))
# TRAIN_SEEDS=($(seq 0 15))
# VALIDATION_SEEDS=($(seq 16 19))

## Launch primitive jobs.

# Critics trained with mean squared regression.

# Pick w/out collisions.
# DATA_CHECKPOINT_PATH="models/20230116/datasets"
# PRIMITIVE="pick"
# CRITIC_CHECKPOINT_PATH="models/20230120/value"
# CRITIC_CHECKPOINTS=("pick_value_sched-cos_iter-2M_sac_ens_value/final_model")
# run_policy

# Place w/out collisions.
# DATA_CHECKPOINT_PATH="models/20230116/datasets"
# PRIMITIVE="place"
# CRITIC_CHECKPOINT_PATH="models/20230120/value"
# CRITIC_CHECKPOINTS=("place_value_sched-cos_iter-5M_sac_ens_value/final_model")
# run_policy

# Pull w/out collisions.
# DATA_CHECKPOINT_PATH="models/20230119/datasets"
# PRIMITIVE="pull"
# CRITIC_CHECKPOINT_PATH="models/20230120/value"
# CRITIC_CHECKPOINTS=("pull_value_sched-cos_iter-2M_sac_ens_value/final_model")
# run_policy

# Push w/out collisions.
# DATA_CHECKPOINT_PATH="models/20230119/datasets"
# PRIMITIVE="push"
# CRITIC_CHECKPOINT_PATH="models/20230120/value"
# CRITIC_CHECKPOINTS=("push_value_sched-cos_iter-2M_sac_ens_value/final_model")
# run_policy

# Critics trained with logistics regression.

# Pick w/out collisions.
# DATA_CHECKPOINT_PATH="models/20230116/datasets"
# PRIMITIVE="pick"
# CRITIC_CHECKPOINT_PATH="models/20230122/value"
# CRITIC_CHECKPOINTS=("pick_value_sched-cos_iter-2M_sac_ens_value_logistics/final_model")
# run_policy

# Place w/out collisions.
# DATA_CHECKPOINT_PATH="models/20230116/datasets"
# PRIMITIVE="place"
# CRITIC_CHECKPOINT_PATH="models/20230122/value"
# CRITIC_CHECKPOINTS=("place_value_sched-cos_iter-5M_sac_ens_value_logistics/final_model")
# run_policy

# Pull w/out collisions.
# DATA_CHECKPOINT_PATH="models/20230119/datasets"
# PRIMITIVE="pull"
# CRITIC_CHECKPOINT_PATH="models/20230122/value"
# CRITIC_CHECKPOINTS=("pull_value_sched-cos_iter-2M_sac_ens_value_logistics/final_model")
# run_policy

# Push w/out collisions.
# DATA_CHECKPOINT_PATH="models/20230119/datasets"
# PRIMITIVE="push"
# CRITIC_CHECKPOINT_PATH="models/20230122/value"
# CRITIC_CHECKPOINTS=("push_value_sched-cos_iter-2M_sac_ens_value_logistics/final_model")
# run_policy
