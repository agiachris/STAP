#!/bin/bash
source ~/.bashrc
source ~/.zshrc
conda activate temporal_policies

# Uniform Sampling
# Easy
python scripts/visualize/visualize_pybox2d.py \
    --exec-config configs/pybox2d/exec/placeright2d_pushleft2d/uniform_sampling.yaml \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/easy/placeright2d_sac/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/easy/pushleft2d_sac/best_model.pt \
    --path ${OUTPUTS}/temporal_policies/visuals/easy/uniform_sampling_multi_step \
    --num-eps 30 \
    --gifs --plot-2d --plot-3d

# Hard
python scripts/visualize/visualize_pybox2d.py \
    --exec-config configs/pybox2d/exec/placeright2d_pushleft2d/uniform_sampling.yaml \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/hard/placeright2d_sac_rand/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/hard/pushleft2d_sac_rand/best_model.pt \
    --path ${OUTPUTS}/temporal_policies/visuals/hard/uniform_sampling_multi_step \
    --num-eps 30 \
    --gifs --plot-2d --plot-3d

# Policy CEM
# Hard
python scripts/visualize/visualize_pybox2d.py \
    --exec-config configs/pybox2d/exec/placeright2d_pushleft2d/policy_cem.yaml \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/hard/placeright2d_sac_rand/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/hard/pushleft2d_sac_rand/best_model.pt \
    --path ${OUTPUTS}/temporal_policies/visuals/hard/policy_cem_multi_step \
    --num-eps 30 \
    --gifs --plot-2d --plot-3d
