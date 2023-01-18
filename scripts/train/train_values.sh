#!/bin/bash

set -e

# GCP_LOGIN="juno-login-lclbjqwy-001"
# GCP_LOGIN="gcp-login-yq0fvtuw-001"

function run_cmd {
    echo ""
    echo "${CMD}"
    if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == juno* ]]; then
        sbatch scripts/train/train_juno.sh "${CMD}"
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
    args="${args} --train-data-checkpoints ${TRAIN_DATA_CHECKPOINTS}"
    args="${args} --eval-data-checkpoints ${EVAL_DATA_CHECKPOINTS}"
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"
    if [ ! -z "${ENCODER_CHECKPOINT}" ]; then
        args="${args} --encoder-checkpoint ${ENCODER_CHECKPOINT}"
    fi
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

# Setup.
DEBUG=0
output_path="models"
plots_path="plots"

# Experiments.
exp_name="20230117/value"
AGENT_CONFIG="configs/pybullet/agents/multi_stage/sac_value.yaml"

# exp_name="20230117/value_ens"
# AGENT_CONFIG="configs/pybullet/agents/multi_stage/sac_ens_value.yaml"

# Pybullet.
TRAINER_CONFIG="configs/pybullet/trainers/value.yaml"
if [[ `hostname` == "sc.stanford.edu" ]] || [[ `hostname` == "${GCP_LOGIN}" ]] || [[ `hostname` == juno* ]]; then
    ENV_KWARGS="--gui 0"
fi

VALUE_OUTPUT_PATH="${output_path}/${exp_name}"

symbolic_action_type="valid"
primitives=("pick" "place" "pull" "push")
train_seeds=("0" "1" "2" "3" "4" "5" "6" "7")
validation_seeds=("8" "9")
data_checkpoint_path="models/20230116/datasets"

# for primitive in "${primitives[@]}"; do
#     ENV_CONFIG="configs/pybullet/envs/t2m/official/primitives/primitives_rl/${primitive}.yaml"

#     TRAIN_DATA_CHECKPOINTS=""
#     for seed in "${train_seeds[@]}"; do
#         data_path="${data_checkpoint_path}/train_${symbolic_action_type}_${primitive}_${seed}/train_data"
#         TRAIN_DATA_CHECKPOINTS="${TRAIN_DATA_CHECKPOINTS} ${data_path}"
#     done

#     for seed in "${validation_seeds[@]}"; do
#         data_path="${data_checkpoint_path}/validation_${symbolic_action_type}_${primitive}_${seed}/train_data"
#         EVAL_DATA_CHECKPOINTS="${EVAL_DATA_CHECKPOINTS} ${data_path}"
#     done

#     train_value
# done

# Sweeps.
primitives=("place")

trainer_config_path="configs/pybullet/trainers/value_sweeps"
trainer_sweeps=(
    "value_l2-0.0001"
    "value_l2-0.001"
    "value_l2-0.01"
    "value_l2-0.1"
)

agent_config_path="configs/pybullet/agents/multi_stage/value_sweeps"
agent_sweeps=(
    "sac_value_hids-2_dims-1024"
    "sac_value_hids-2_dims-512"
    "sac_value_hids-3_dims-1024"
    "sac_value_hids-3_dims-256"
    "sac_value_hids-3_dims-512"
    "sac_value_hids-4_dims-1024"
    "sac_value_hids-4_dims-256"
    "sac_value_hids-4_dims-512"
)

for primitive in "${primitives[@]}"; do
    ENV_CONFIG="configs/pybullet/envs/t2m/official/primitives/primitives_rl/${primitive}.yaml"
    
    for trainer_name in "${trainer_sweeps[@]}"; do
        TRAINER_CONFIG="${trainer_config_path}/${trainer_name}.yaml"
    
        for agent_name in "${agent_sweeps[@]}"; do
            AGENT_CONFIG="${agent_config_path}/${agent_name}.yaml"

            TRAIN_DATA_CHECKPOINTS=""
            for seed in "${train_seeds[@]}"; do
                data_path="${data_checkpoint_path}/train_${symbolic_action_type}_${primitive}_${seed}/train_data"
                TRAIN_DATA_CHECKPOINTS="${TRAIN_DATA_CHECKPOINTS} ${data_path}"
            done

            for seed in "${validation_seeds[@]}"; do
                data_path="${data_checkpoint_path}/validation_${symbolic_action_type}_${primitive}_${seed}/train_data"
                EVAL_DATA_CHECKPOINTS="${EVAL_DATA_CHECKPOINTS} ${data_path}"
            done

            NAME="${primitive}/${trainer_name}/${agent_name}"
            train_value
        done
    done
done
