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

function train_value {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --agent-config ${AGENT_CONFIG}"
    args="${args} --env-config ${ENV_CONFIG}"
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
        args="${args} --path ${VALUE_OUTPUT_PATH}_debug"
        args="${args} --overwrite"
        args="${args} --num-train-steps 10"
        args="${args} --num-eval-episodes 10"
    else
        args="${args} --path ${VALUE_OUTPUT_PATH}"
    fi

    CMD="python scripts/train/train_policy.py ${args}"
    run_cmd
}

function run_value {
    ENV_CONFIG="configs/pybullet/envs/t2m/official/primitives/primitives_rl/${PRIMITIVE}.yaml"

    TRAIN_DATA_CHECKPOINTS=""
    for seed in "${TRAIN_SEEDS[@]}"; do
        data_path="${DATA_CHECKPOINT_PATH}/train_${SYMBOLIC_ACTION_TYPE}_${PRIMITIVE}_${seed}/train_data"
        TRAIN_DATA_CHECKPOINTS="${TRAIN_DATA_CHECKPOINTS} ${data_path}"
    done

    for seed in "${VALIDATION_SEEDS[@]}"; do
        data_path="${DATA_CHECKPOINT_PATH}/validation_${SYMBOLIC_ACTION_TYPE}_${PRIMITIVE}_${seed}/train_data"
        EVAL_DATA_CHECKPOINTS="${EVAL_DATA_CHECKPOINTS} ${data_path}"
    done

    train_value
}

### Setup.
SBATCH_SLURM="scripts/train/train_juno.sh"
DEBUG=0
output_path="models"
plots_path="plots"

### Experiments.

## Pybullet.
exp_name="20230118/value"
VALUE_OUTPUT_PATH="${output_path}/${exp_name}"

TRAINER_CONFIG="configs/pybullet/trainers/value.yaml"
AGENT_CONFIG="configs/pybullet/agents/multi_stage/sac_ens_value.yaml"
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]] || [[ `hostname` == juno* ]]; then
    ENV_KWARGS="--gui 0"
fi

# Data.
SYMBOLIC_ACTION_TYPE="valid"
TRAIN_SEEDS=("0" "1" "2" "3" "4" "5" "6" "7")
VALIDATION_SEEDS=("8" "9")
DATA_CHECKPOINT_PATH="models/20230116/datasets"

# Launch primitive jobs.
PRIMITIVE="pick"
run_value

PRIMITIVE="place"
run_value

PRIMITIVE="pull"
run_value

PRIMITIVE="push"
run_value


# Sweeps.
function run_value_sweep {    
    for trainer_name in "${TRAINER_SWEEPS[@]}"; do
        TRAINER_CONFIG="${TRAINER_CONFIG_PATH}/${trainer_name}.yaml"
    
        for agent_name in "${AGENT_SWEEPS[@]}"; do
            AGENT_CONFIG="${AGENT_CONFIG_PATH}/${agent_name}.yaml"

            if [[ "${agent_name}" == *dims-256 ]]; then
                SBATCH_SLURM="scripts/train/train_juno_cpu.sh"
            else
                SBATCH_SLURM="scripts/train/train_juno.sh"
            fi

            NAME="${PRIMITIVE}_${trainer_name}_${agent_name}"
            run_value
        done
    done
}

# Set sweep config paths.
TRAINER_CONFIG_PATH="configs/pybullet/trainers/value_sweeps"
TRAINER_SWEEPS=(
    "value_l2-0.0001"
    "value_l2-0.001"
    "value_l2-0.01"
    "value_l2-0.1"
)

AGENT_CONFIG_PATH="configs/pybullet/agents/multi_stage/value_sweeps"
AGENT_SWEEPS=(
    "sac_value_hids-2_dims-1024"
    "sac_value_hids-2_dims-512"
    "sac_value_hids-3_dims-1024"
    "sac_value_hids-3_dims-256"
    "sac_value_hids-3_dims-512"
    "sac_value_hids-4_dims-1024"
    "sac_value_hids-4_dims-256"
    "sac_value_hids-4_dims-512"
)

# Launch primitive sweep jobs.
# PRIMITIVE="pick"
# run_value_sweep

# PRIMITIVE="place"
# run_value_sweep

# PRIMITIVE="pull"
# run_value_sweep

# PRIMITIVE="push"
# run_value_sweep
