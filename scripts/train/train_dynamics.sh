#!/bin/bash

#python scripts/train.py -c configs/pybox2d/placeright2d/sac.yaml -p models/placeright2d
#python scripts/train.py -c configs/pybox2d/pushleft2dcontrol/sac.yaml -p models/pushleft2dcontrol
#python scripts/train.py -c configs/pybox2d/placeright2d/sac_rand.yaml -p models/placeright2d_rand
#python scripts/train.py -c configs/pybox2d/pushleft2dcontrol/sac_rand.yaml -p models/pushleft2dcontrol_rand

#python scripts/train/train_dynamics.py -c configs/pybox2d/dynamics/decoupled.yaml --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/policy.yaml -p models/placeright2d_pushleft2dcontrol --policy-checkpoints models/placeright2d/final_model.pt models/pushleft2dcontrol/final_model.pt
python scripts/train/train_dynamics.py -c configs/pybox2d/dynamics/decoupled.yaml --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/policy.yaml -p models/placeright2d_pushleft2dcontrol_rand --policy-checkpoints models/placeright2d_rand/final_model.pt models/pushleft2dcontrol_rand/final_model.pt
