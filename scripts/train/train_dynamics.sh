#!/bin/bash

python scripts/train.py -c configs/pybox2d/placeright2d/sac.yaml -p models/placeright2d
python scripts/train.py -c configs/pybox2d/pushleft2dcontrol/sac.yaml -p models/pushleft2dcontrol
python scripts/train.py -c configs/pybox2d/placeright2d/sac_rand.yaml -p models/placeright2d_rand
python scripts/train.py -c configs/pybox2d/pushleft2dcontrol/sac_rand.yaml -p models/pushleft2dcontrol_rand

python scripts/train/train_dynamics.py -c configs/pybox2d/dynamics/decoupled.yaml --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/policy.yaml -p models/placeright2d_pushleft2dcontrol --checkpoints models/placeright2d/best_model.pt models/pushleft2dcontrol/best_model.pt
python scripts/train/train_dynamics.py -c configs/pybox2d/dynamics/decoupled.yaml --exec-config configs/pybox2d/exec/placeright2d_pushleft2dcontrol/policy.yaml -p models/placeright2d_pushleft2dcontrol_rand --checkpoints models/placeright2d_rand/best_model.pt models/pushleft2dcontrol_rand/best_model.pt
