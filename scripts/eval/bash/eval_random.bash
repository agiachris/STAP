#!/bin/bash
source ~/.bashrc
source ~/.zshrc
conda activate temporal_policies

# Easy
python scripts/visualize/primitives_pybox2d.py \
    --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/random_policy.yaml \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/easy/placeright2d_sac/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/easy/pushleft2dcontrol_sac/best_model.pt \
    --num-eps 100 

# Hard
python scripts/visualize/primitives_pybox2d.py \
    --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/random_policy.yaml \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/hard/placeright2d_sac_rand/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/hard/pushleft2dcontrol_sac_rand/best_model.pt \
    --num-eps 100
