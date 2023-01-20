#!/bin/bash

set -e

GCP_LOGIN="juno-login-lclbjqwy-001"

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

function train_dynamics {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --dynamics-config ${DYNAMICS_CONFIG}"
    if [ ${#POLICY_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS[@]}"
    fi
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"

    if [ ! -z "${NAME}" ]; then
        args="${args} --name ${NAME}"
    fi
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path ${DYNAMICS_OUTPUT_PATH}_debug"
        args="${args} --overwrite"
        args="${args} --num-train-steps 10000"
    else
        args="${args} --path ${DYNAMICS_OUTPUT_PATH}"
    fi

    CMD="python scripts/train/train_dynamics.py ${args}"
    run_cmd
}

function run_dynamics {
    NAME=""
    POLICY_CHECKPOINTS=()
    for primitive in "${PRIMITIVES[@]}"; do
        POLICY_CHECKPOINTS+=("${POLICY_CHECKPOINT_PATH}/${primitive}/${POLICY_DIRS[${primitive}]}/final_model.pt")

        if [ -z "${NAME}" ]; then
            NAME="${primitive}"
        else
            NAME="${NAME}_${primitive}"
        fi
    done
    NAME="${NAME}_dynamics"

    train_dynamics
}

### Setup.
SBATCH_SLURM="scripts/train/train_juno.sh"
DEBUG=0
output_path="models"

### Experiments.

## Pybullet.
exp_name="20230119/dynamics"
DYNAMICS_OUTPUT_PATH="${output_path}/${exp_name}"

DYNAMICS_CONFIG="configs/pybullet/dynamics/table_env.yaml"
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]] || [[ `hostname` == juno* ]]; then
    ENV_KWARGS="--gui 0"
fi

# Launch dynamics jobs.

# Pick dynamics.
TRAINER_CONFIG="configs/pybullet/trainers/dynamics_primitive.yaml"
POLICY_CHECKPOINT_PATH="models/20230119/policy"
PRIMITIVES=("pick")
declare -A POLICY_DIRS=(
    ["pick"]="final_model"
)
run_dynamics

# Place dynamics.
TRAINER_CONFIG="configs/pybullet/trainers/dynamics_primitive.yaml"
POLICY_CHECKPOINT_PATH="models/20230119/policy"
PRIMITIVES=("place")
declare -A POLICY_DIRS=(
    ["place"]="final_model"
)
run_dynamics

# Pull dynamics.
TRAINER_CONFIG="configs/pybullet/trainers/dynamics_primitive.yaml"
POLICY_CHECKPOINT_PATH="models/20230119/policy"
PRIMITIVES=("pull")
declare -A POLICY_DIRS=(
    ["pull"]="final_model"
)
run_dynamics

# Push dynamics.
TRAINER_CONFIG="configs/pybullet/trainers/dynamics_primitive.yaml"
POLICY_CHECKPOINT_PATH="models/20230119/policy"
PRIMITIVES=("push")
declare -A POLICY_DIRS=(
    ["push"]="final_model"
)
run_dynamics

# Full suite dynamics.
TRAINER_CONFIG="configs/pybullet/trainers/dynamics.yaml"
POLICY_CHECKPOINT_PATH="models/20230119/policy"
PRIMITIVES=(
    "pick"
    "place"
    "pull"
    "push"
)
declare -A POLICY_DIRS=(
    ["pick"]="final_model"
    ["place"]="final_model"
    ["pull"]="final_model"
    ["push"]="final_model"
)
run_dynamics
