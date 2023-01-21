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

### Setup.
SBATCH_SLURM="scripts/train/train_juno.sh"
DEBUG=0
output_path="models"

### Experiments.

## Pybullet.
exp_name="20230121/dynamics"
DYNAMICS_OUTPUT_PATH="${output_path}/${exp_name}"

DYNAMICS_CONFIG="configs/pybullet/dynamics/table_env.yaml"
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]] || [[ `hostname` == juno* ]]; then
    ENV_KWARGS="--gui 0"
fi

# Launch primitive dynamics jobs.
TRAINER_CONFIG="configs/pybullet/trainers/dynamics_primitive.yaml"
PRIMITIVES=("pick")
declare -A POLICY_CHECKPOINT_PATHS=(["pick"]="models/20230121/policy/pick_value_sched-cos_iter-2M_sac_ens_value/final_model/final_model.pt")
run_dynamics

TRAINER_CONFIG="configs/pybullet/trainers/dynamics_primitive.yaml"
PRIMITIVES=("place")
declare -A POLICY_CHECKPOINT_PATHS=(["place"]="models/20230121/policy/place_value_sched-cos_iter-5M_sac_ens_value/final_model/final_model.pt")
run_dynamics

TRAINER_CONFIG="configs/pybullet/trainers/dynamics_primitive.yaml"
PRIMITIVES=("pull")
declare -A POLICY_CHECKPOINT_PATHS=(["pull"]="models/20230120/policy/pull_value_sched-cos_iter-2M_sac_ens_value/final_model/final_model.pt")
run_dynamics

TRAINER_CONFIG="configs/pybullet/trainers/dynamics_primitive.yaml"
PRIMITIVES=("push")
declare -A POLICY_CHECKPOINT_PATHS=(["push"]="models/20230120/policy/push_value_sched-cos_iter-2M_sac_ens_value/final_model/final_model.pt")
run_dynamics

# Launch full suite dynamics jobs.
TRAINER_CONFIG="configs/pybullet/trainers/dynamics.yaml"
PRIMITIVES=(
    "pick"
    "place"
    "pull"
    "push"
)
declare -A POLICY_CHECKPOINT_PATHS=(
    ["pick"]="models/20230121/policy/pick_value_sched-cos_iter-2M_sac_ens_value/final_model/final_model.pt"
    ["place"]="models/20230121/policy/place_value_sched-cos_iter-5M_sac_ens_value/final_model/final_model.pt"
    ["pull"]="models/20230120/policy/pull_value_sched-cos_iter-2M_sac_ens_value/final_model/final_model.pt"
    ["push"]="models/20230120/policy/push_value_sched-cos_iter-2M_sac_ens_value/final_model/final_model.pt"
)
run_dynamics
