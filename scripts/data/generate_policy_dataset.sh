#!/bin/bash

# Set the base command
base_command="PYTHONPATH=. python scripts/data/generate_policy_dataset.py \
--config.pddl-cfg.pddl-domain template \
--config.num-pretrain-steps 50000 --config.num-train-steps 0 --config.num-eval-episodes 0 \
--config.exp-name 20230105/dataset_collection/ \
--config.min-num-box-obj 3 --config.max-num-box-obj 4 \
--config.primitive"

echo "Usage of scripts/data/generate_policy_dataset.py"
eval PYTHONPATH=. python scripts/data/generate_policy_dataset.py -h

# Set the primitives and seeds
primitives=("pick") # "place" "push" "pull")
final_seeds=("40")

# Loop through the primitives and seeds (go from 0 to the respective final seed, non-inclusive) and run the command
for i in "${!primitives[@]}"; do
    for seed in $(seq 0 $((${final_seeds[$i]} - 1))); do
        command="${base_command} ${primitives[$i]} --config.seed ${seed}"
        echo $command
        eval $command
    done
done
