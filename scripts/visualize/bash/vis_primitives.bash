#!/bin/bash
source ~/.bashrc
source ~/.zshrc
conda activate temporal_policies

# Random Policy
# Easy
python scripts/visualize/visualize_pybox2d.py \
    --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/random_policy.yaml \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/easy/placeright2d_sac/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/easy/pushleft2dcontrol_sac/best_model.pt \
    --path ${OUTPUTS}/temporal_policies/visuals/easy/random_policy \
    --num-eps 30 \
    --gifs

# Hard
python scripts/visualize/visualize_pybox2d.py \
    --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/random_policy.yaml \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/hard/placeright2d_sac_rand/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/hard/pushleft2dcontrol_sac_rand/best_model.pt \
    --path ${OUTPUTS}/temporal_policies/visuals/hard/random_policy \
    --num-eps 30 \
    --gifs

# Policy
# Easy
python scripts/visualize/visualize_pybox2d.py \
    --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/policy.yaml \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/easy/placeright2d_sac/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/easy/pushleft2dcontrol_sac/best_model.pt \
    --path ${OUTPUTS}/temporal_policies/visuals/easy/policy \
    --num-eps 30 \
    --gifs --plot-2d --plot-3d

# Hard
python scripts/visualize/visualize_pybox2d.py \
    --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/policy.yaml \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/hard/placeright2d_sac_rand/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/hard/pushleft2dcontrol_sac_rand/best_model.pt \
    --path ${OUTPUTS}/temporal_policies/visuals/hard/policy \
    --num-eps 30 \
    --gifs --plot-2d --plot-3d

# Uniform Sampling
# Easy
python scripts/visualize/visualize_pybox2d.py \
    --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/uniform_sampling.yaml \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/easy/placeright2d_sac/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/easy/pushleft2dcontrol_sac/best_model.pt \
    --path ${OUTPUTS}/temporal_policies/visuals/easy/uniform_sampling \
    --num-eps 30 \
    --gifs --plot-2d --plot-3d

# Hard
python scripts/visualize/visualize_pybox2d.py \
    --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/uniform_sampling.yaml \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/hard/placeright2d_sac_rand/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/hard/pushleft2dcontrol_sac_rand/best_model.pt \
    --path ${OUTPUTS}/temporal_policies/visuals/hard/uniform_sampling \
    --num-eps 30 \
    --gifs --plot-2d --plot-3d

# Policy CEM
# Hard
python scripts/visualize/visualize_pybox2d.py \
    --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/policy_cem.yaml \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/hard/placeright2d_sac_rand/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/hard/pushleft2dcontrol_sac_rand/best_model.pt \
    --path ${OUTPUTS}/temporal_policies/visuals/hard/policy_cem \
    --num-eps 30 \
    --gifs --plot-2d --plot-3d
