#!/bin/bash
source ~/.bashrc
source ~/.zshrc
conda activate temporal_policies

# Basic 
python scripts/eval/eval_pybox2d.py \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/basic/PlaceRight2D_SAC_BASIC/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/basic/PushLeft2D_SAC_BASIC/best_model.pt \
    --num-eps 1000 \
    --eval-planner random_sampling \
    --num-samples 100

# Rand
python scripts/eval/eval_pybox2d.py \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/rand/PlaceRight2D_SAC_RAND/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/rand/PushLeft2D_SAC_RAND/best_model.pt \
    --num-eps 1000 \
    --eval-planner random_sampling \
    --num-samples 100
