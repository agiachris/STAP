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

function eval_tamp {
    args=""
    args="${args} --planner-config ${PLANNER_CONFIG}"
    args="${args} --env-config ${ENV_CONFIG}"
    if [ ${#POLICY_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS[@]}"
    fi
    if [ ${#SCOD_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --scod-checkpoints ${SCOD_CHECKPOINTS[@]}"
    fi
    if [ ! -z "${DYNAMICS_CHECKPOINT}" ]; then
        args="${args} --dynamics-checkpoint ${DYNAMICS_CHECKPOINT}"
    fi
    args="${args} --seed 0"
    args="${args} --pddl-domain ${PDDL_DOMAIN}"
    args="${args} --pddl-problem ${PDDL_PROBLEM}"
    args="${args} --max-depth 4"
    args="${args} --timeout 10"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --num-eval 10"
        args="${args} --path ${PLANNER_OUTPUT_PATH}_debug"
        args="${args} --verbose 1"
    else
        args="${args} --num-eval 100"
        args="${args} --path ${PLANNER_OUTPUT_PATH}"
        args="${args} --verbose 0"
    fi
    CMD="python scripts/eval/eval_tamp.py ${args}"
    run_cmd
}

function run_tamp {
    for task in "${TASKS[@]}"; do
        PLANNER_OUTPUT_PATH="${PLANNER_OUTPUT_ROOT}/${task}"
        ENV_CONFIG="${TASK_ROOT}/${task}.yaml"
        PDDL_DOMAIN="${TASK_ROOT}/${task}_domain.pddl"
        PDDL_PROBLEM="${TASK_ROOT}/${task}_problem.pddl"

        for planner in "${PLANNERS[@]}"; do
            PLANNER_CONFIG="${PLANNER_CONFIG_PATH}/${planner}.yaml"

            POLICY_CHECKPOINTS=()
            for primitive in "${PRIMITIVES[@]}"; do
                POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${primitive}/${CHECKPOINT}.pt")
            done

            SCOD_CHECKPOINTS=()
            if [[ "${planner}" == *scod* ]]; then
                for primitive in "${PRIMITIVES[@]}"; do
                    SCOD_CHECKPOINTS+=("${SCOD_INPUT_PATH}/${CHECKPOINT}/${primitive}/final_scod.pt")
                done
            fi

            if [[ "${planner}" == *_oracle_*dynamics ]]; then
                DYNAMICS_CHECKPOINT=""
            else
                DYNAMICS_CHECKPOINT="${DYNAMICS_INPUT_PATH}/${CHECKPOINT}.pt"
            fi

            eval_tamp
        done
    done
}

# Evaluation tasks: Uncomment tasks to evaluate.
TASK_ROOT="configs/pybullet/envs/official/sim_domains"
TASKS=(
    "hook_reach/tamp0"
    "constrained_packing/tamp0"
    "rearrangement_push/tamp0"
)

# Planners: Uncomment planners to evaluate.
PLANNER_CONFIG_PATH="configs/pybullet/planners"
PLANNERS=(
    "ablation/policy_cem"
    # "ablation/scod_policy_cem"
    "ablation/policy_shooting"
    "ablation/random_cem"
    "ablation/random_shooting"
    "greedy"
)

# Setup.
SBATCH_SLURM="scripts/eval/eval_planners_juno.sh"
DEBUG=0

input_path="models"
output_path="plots"

# Pybullet experiments.
if [[ `hostname` == *stanford.edu ]] || [[ `hostname` == juno* ]]; then
    ENV_KWARGS="--gui 0"
fi
ENV_KWARGS="${ENV_KWARGS} --closed-loop 1"

# Evaluate tamp.
exp_name="tamp"
PLANNER_OUTPUT_ROOT="${output_path}/${exp_name}"

PRIMITIVES=(
    "pick"
    "place"
    "pull"
    "push"
)
CHECKPOINT="official_model"
POLICY_INPUT_PATH="${input_path}/agents_rl"
DYNAMICS_INPUT_PATH="${input_path}/dynamics_rl/pick_place_pull_push_dynamics"
SCOD_INPUT_PATH="${input_path}/scod_rl"
run_tamp