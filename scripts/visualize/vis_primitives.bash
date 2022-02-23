#!/bin/bash
source ~/.bashrc
source ~/.zshrc
conda activate temporal_policies

# Basic
python scripts/visualize/primitives_pybox2d.py \
    --path ${OUTPUTS}/temporal_policies/visuals/gifs/SAC_BASIC \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/basic/PlaceRight2D_SAC_BASIC/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/basic/PushLeft2D_SAC_BASIC/best_model.pt \
    --num-eps 100 \
    --vis-random

# Rand
python scripts/visualize/primitives_pybox2d.py \
    --path ${OUTPUTS}/temporal_policies/visuals/gifs/SAC_RAND \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/rand/PlaceRight2D_SAC_RAND/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/rand/PushLeft2D_SAC_RAND/best_model.pt \
    --num-eps 100 \
    --vis-random
