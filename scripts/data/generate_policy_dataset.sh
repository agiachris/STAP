#!/bin/bash

# Set the base command
base_command="PYTHONPATH=. python scripts/data/generate_policy_dataset.py \
--config.num-pretrain-steps 10 --config.num-train-steps 0 --config.num-eval-episodes 0 \
--config.primitive"

echo "Usage of scripts/data/generate_policy_dataset.py"
eval PYTHONPATH=. python scripts/data/generate_policy_dataset.py -h

# Set the primitives and seeds
primitives=("pick" "place" "push" "pull")
final_seeds=("4" "6" "5" "5")

# Loop through the primitives and seeds (go from 0 to the respective final seed, non-inclusive) and run the command
for i in "${!primitives[@]}"; do
    for seed in $(seq 0 $((${final_seeds[$i]} - 1))); do
        command="${base_command} ${primitives[$i]} --config.seed ${seed}"
        echo $command
        eval $command
    done
done
