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
        POLICY_CHECKPOINTS+=("${POLICY_CHECKPOINT_PATHS[${primitive}]}")

        if [ -z "${NAME}" ]; then
            NAME="${primitive}"
        else
            NAME="${NAME}_${primitive}"
        fi
    done
    NAME="${NAME}_dynamics"

    train_dynamics
}

# Setup.
SBATCH_SLURM="scripts/train/train_juno.sh"
DEBUG=0

output_path="models"
exp_name="20230309/dynamics"
DYNAMICS_OUTPUT_PATH="${output_path}/${exp_name}"
DYNAMICS_CONFIG="configs/pybullet/dynamics/table_env.yaml"

if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]] || [[ `hostname` == juno* ]]; then
    ENV_KWARGS="--gui 0"
fi

# Launch full suite dynamics jobs.
TRAINER_CONFIG="configs/pybullet/trainers/dynamics/dynamics_iter-0.75M.yaml"
PRIMITIVES=(
    "pick"
    "place"
    "pull"
    "push"
)
declare -A POLICY_CHECKPOINT_PATHS=(
    ["pick"]="models/20230309/policy/pick/final_model/final_model.pt"
    ["place"]="models/20230309/policy/place/final_model/final_model.pt"
    ["pull"]="models/20230309/policy/pull/final_model/final_model.pt"
    ["push"]="models/20230309/policy/push/final_model/final_model.pt"
)
run_dynamics
