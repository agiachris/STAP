#!/bin/bash

set -e

# GCP_LOGIN="juno-login-lclbjqwy-001"
GCP_LOGIN="gcp-login-yq0fvtuw-001"

function run_cmd {
    echo ""
    echo "${CMD}"
    if [[ `hostname` == "sc.stanford.edu" ]]; then
        sbatch scripts/train/train_juno.sh "${CMD}"
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
    if [ ! -z "${EVAL_ENV_CONFIG}" ]; then
        args="${args} --eval-env-config ${EVAL_ENV_CONFIG}"
    fi
    if [ ! -z "${ENCODER_CHECKPOINT}" ]; then
        args="${args} --encoder-checkpoint ${ENCODER_CHECKPOINT}"
    fi
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"
    args="${args} --train-multistage 1"
    args="${args} --num-critic-only-train-steps 150000"
    args="${args} --num-actor-only-train-steps 150000"
    args="${args} --num-original-train-steps 50000"

    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path ${POLICY_OUTPUT_PATH}_debug"
        # args="${args} --eval-recording-path ${EVAL_RECORDING_PATH}_debug"
        args="${args} --overwrite"
    else
        args="${args} --path ${POLICY_OUTPUT_PATH}"
        args="${args} --eval-recording-path ${EVAL_RECORDING_PATH}"
    fi

    CMD="python scripts/train/train_policy.py ${args}"
    run_cmd
}

# Setup.
DEBUG=0
output_path="models"
plots_path="plots"

# Experiments.

primitives=($1)
echo "primitives: ${primitives[@]}"

# loop through the primitives and train the policy
for primitive in "${primitives[@]}"; do
    exp_name="20230101/complete_q_multistage_${primitive}"
    AGENT_CONFIG="configs/pybullet/agents/sac_multistage.yaml"

    TRAINER_CONFIG="configs/pybullet/trainers/agent_multistage_${primitive}.yaml"

    POLICY_OUTPUT_PATH="${output_path}/${exp_name}"
    EVAL_RECORDING_PATH="${plots_path}/${exp_name}"
    ENV_KWARGS="--num-env-processes 4 --num-eval-env-processes 1 --gui 0"
    if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]]; then
        ENV_KWARGS="${ENV_KWARGS} --gui 0"
    fi

    ENV_CONFIG="configs/pybullet/envs/official/primitives/${primitive}_0_symbolically_valid_actions.yaml"
    EVAL_ENV_CONFIG="configs/pybullet/envs/official/primitives/${primitive}_0_symbolically_valid_actions.yaml"
    # train_policy
done

# primitives=("pick" "place" "push" "pull")
# # loop through the primitives and train the policy
# for primitive in "${primitives[@]}"; do
#     exp_name="20230101/complete_q_multistage_${primitive}"
#     AGENT_CONFIG="configs/pybullet/agents/sac_multistage.yaml"

#     TRAINER_CONFIG="configs/pybullet/trainers/agent_multistage_${primitive}.yaml"

#     POLICY_OUTPUT_PATH="${output_path}/${exp_name}"
#     EVAL_RECORDING_PATH="${plots_path}/${exp_name}"
#     ENV_KWARGS="--num-env-processes 4 --num-eval-env-processes 1 --gui 0"
#     if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]]; then
#         ENV_KWARGS="${ENV_KWARGS} --gui 0"
#     fi

#     ENV_CONFIG="configs/pybullet/envs/official/primitives/${primitive}_0_symbolically_valid_actions.yaml"
#     EVAL_ENV_CONFIG="configs/pybullet/envs/official/primitives/${primitive}_0_symbolically_valid_actions.yaml"
#     train_policy
# done
