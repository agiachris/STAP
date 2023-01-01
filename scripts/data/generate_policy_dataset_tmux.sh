#!/bin/bash

# Set the base command
base_command="PYTHONPATH=. python scripts/data/generate_policy_dataset.py \
--config.num-pretrain-steps 100000 --config.num-train-steps 0 --config.num-eval-episodes 0 --config.gui 0 \
--config.primitive"

echo "Please input the name of the conda environment you want to use for each command"
echo "Usage of scripts/data/generate_policy_dataset.py"
eval PYTHONPATH=. python scripts/data/generate_policy_dataset.py -h

# Set the primitives and seeds
primitives=("pick" "place" "pull" "push")
start_seeds=("0" "0" "0" "0")
final_seeds=("8" "12" "10" "10")

for i in "${!primitives[@]}"; do
    for seed in $(seq ${start_seeds[$i]} $((${final_seeds[$i]} - 1))); do
        session_name="primitive-${primitives[$i]}-seed-${seed}"
        command="conda activate $1 && ${base_command} ${primitives[$i]} --config.seed ${seed}"
        echo $command
        tmux new-session -s "$session_name" -d
        tmux send-keys -t "$session_name" "$command" C-m
    done
done

# for each primitive in the list of primitives (pick, place, push, pull) 
# train run the command for the seed 100
for primitive in "${primitives[@]}"; do
    session_name="primitive-${primitive}-seed-100"
    command="conda activate $1 && ${base_command} ${primitive} --config.seed 100"
    echo $command
    tmux new-session -s "$session_name" -d
    tmux send-keys -t "$session_name" "$command" C-m
done
