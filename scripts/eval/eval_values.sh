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


#### Setup.
SBATCH_SLURM="scripts/train/train_juno.sh"
DEBUG=0
output_path="plots"

#### Experiments.

### Pybullet.
exp_name="20230126/value_logistics"
VALUE_OUTPUT_PATH="${output_path}/${exp_name}"
NUM_EVAL_STEPS=1000

## Launch primitive jobs.

## Data.
# DATA_CONFIG="configs/pybullet/datasets/replay_buffer.yaml"
# SYMBOLIC_ACTION_TYPE="valid"
# SEEDS=("0")

## Critics trained with mean squared regression.

# DATA_CONFIG="configs/pybullet/datasets/replay_buffer.yaml"
# SEEDS=("0")
# declare -A PRIMITIVES=(
#     ["pick"]="pick_value_sched-cos_iter-2M_sac_ens_value"
#     ["place"]="place_value_sched-cos_iter-5M_sac_ens_value"
#     ["pull"]="pull_value_sched-cos_iter-2M_sac_ens_value"
#     ["push"]="push_value_sched-cos_iter-2M_sac_ens_value"
# )
# declare -A AGENT_CHECKPOINT_PATHS=(
#     ["pick"]="models/20230120/value"
#     ["place"]="models/20230120/value"
#     ["pull"]="models/20230120/value"
#     ["push"]="models/20230120/value"
# )
# declare -A DATA_CHECKPOINT_PATHS=(
#     ["pick"]="models/20230116/datasets"
#     ["place"]="models/20230116/datasets"
#     ["pull"]="models/20230119/datasets"
#     ["push"]="models/20230119/datasets"
# )
# for PRIMITIVE in "${!PRIMITIVES[@]}"; do
#     AGENT_CHECKPOINT_PATH="${AGENT_CHECKPOINT_PATHS[${PRIMITIVE}]}"
    
#     AGENT_CHECKPOINT="${PRIMITIVES[${PRIMITIVE}]}"
#     SYMBOLIC_ACTION_TYPE="valid"
#     DATA_CHECKPOINT_PATH="${DATA_CHECKPOINT_PATHS[${PRIMITIVE}]}"
#     TAG="ind"
#     run_value
    
#     AGENT_CHECKPOINT="${PRIMITIVES[${PRIMITIVE}]}"
#     SYMBOLIC_ACTION_TYPE="invalid"
#     DATA_CHECKPOINT_PATH="models/20230125/datasets"
#     TAG="ood"
#     run_value
# done

## Critics trained with logistics regression, balanced data (40%).

## Data.
DATA_CONFIG="configs/pybullet/datasets/replay_buffer.yaml"
SEEDS=("0")
declare -A AGENT_CHECKPOINT_PATHS=(
    ["pick"]="models/20230124/value"
    ["place"]="models/20230124/value"
    ["pull"]="models/20230124/value"
    ["push"]="models/20230124/value"
)
for PRIMITIVE in "${!AGENT_CHECKPOINT_PATHS[@]}"; do
    AGENT_CHECKPOINT_PATH="${AGENT_CHECKPOINT_PATHS[${PRIMITIVE}]}"
    
    AGENT_CHECKPOINT="${PRIMITIVE}"
    SYMBOLIC_ACTION_TYPE="valid"
    DATA_CHECKPOINT_PATH="models/20230124/datasets"
    TAG="ind"
    run_value
    
    AGENT_CHECKPOINT="${PRIMITIVE}"
    SYMBOLIC_ACTION_TYPE="invalid"
    DATA_CHECKPOINT_PATH="models/20230125/datasets"
    TAG="ood"
    run_value
done

## Critics trained with MSE loss and no sigmoid activation, balanced data (40%).

# DATA_CONFIG="configs/pybullet/datasets/replay_buffer.yaml"
# SEEDS=("0")
# declare -A AGENT_CHECKPOINT_PATHS=(
#     ["pick"]="models/20230126/value"
#     ["place"]="models/20230126/value"
#     ["pull"]="models/20230126/value"
#     ["push"]="models/20230126/value"
# )
# for PRIMITIVE in "${!AGENT_CHECKPOINT_PATHS[@]}"; do
#     AGENT_CHECKPOINT_PATH="${AGENT_CHECKPOINT_PATHS[${PRIMITIVE}]}"
    
#     AGENT_CHECKPOINT="${PRIMITIVE}"
#     SYMBOLIC_ACTION_TYPE="valid"
#     DATA_CHECKPOINT_PATH="models/20230124/datasets"
#     TAG="ind"
#     run_value
    
#     AGENT_CHECKPOINT="${PRIMITIVE}"
#     SYMBOLIC_ACTION_TYPE="invalid"
#     DATA_CHECKPOINT_PATH="models/20230125/datasets"
#     TAG="ood"
#     run_value
# done
