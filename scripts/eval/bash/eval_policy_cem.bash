#!/bin/bash
source ~/.bashrc
source ~/.zshrc
conda activate temporal_policies

# Easy
# python scripts/eval/sweep_pybox2d.py \
#     --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/policy_cem.yaml \
#     --sweep-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/sweeps/policy_cem.json \
#     --checkpoints \
#     ${OUTPUTS}/temporal_policies/models/easy/placeright2d_sac/best_model.pt \
#     ${OUTPUTS}/temporal_policies/models/easy/pushleft2dcontrol_sac/best_model.pt \
#     --path $OUTPUTS/temporal_policies/results/easy \
#     --num-eps 100 

# Hard
python scripts/eval/sweep_pybox2d.py \
    --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/policy_cem.yaml \
    --sweep-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/sweeps/policy_cem.json \
    --checkpoints \
    ${OUTPUTS}/temporal_policies/models/hard/placeright2d_sac_rand/best_model.pt \
    ${OUTPUTS}/temporal_policies/models/hard/pushleft2dcontrol_sac_rand/best_model.pt \
    --path $OUTPUTS/temporal_policies/results/hard \
    --num-eps 100
