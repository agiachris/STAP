#!/bin/bash
source ~/.bashrc
source ~/.zshrc
conda activate temporal_policies

# Easy
python scripts/eval/eval_pybox2d.py \
    --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/random_policy.yaml \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/easy/placeright2d_sac/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/easy/pushleft2dcontrol_sac/best_model.pt \
    --path $OUTPUTS/temporal_policies/results/easy/random_policy.json \
    --num-eps 100 

# Hard
python scripts/eval/eval_pybox2d.py \
    --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/random_policy.yaml \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/hard/placeright2d_sac_rand/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/hard/pushleft2dcontrol_sac_rand/best_model.pt \
    --path $OUTPUTS/temporal_policies/results/hard/random_policy.json \
    --num-eps 100
