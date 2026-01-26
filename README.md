# Demo Panda Manipulation

This repository shows a simple object manipulation policy using the Franka Emika Panda in NVIDIA Isaac Sim.

## Overview

- Robot: Franka Emika Panda
- Task: Non-randomized object lifting
- Algorithm: Proximal policy optimization
- Simulator: NVIDIA Isaac Sim / IsaacLab

## Directory Structure

```text
├── configs/
│ ├── python
│   ├── env_cfg.py     # Environment configuration class
│   └── scene_cfg.py   # Scene configuration class
│ └── yaml
│   ├── train.yaml     # Configuration for train.py
│   └── inference.yaml # Configuration for inference.py
├── rl/
│   ├── ppo.py         # Actor/critic class definitions
│   └── rollout.py     # Rollout class (transition storage)
├── scripts/
│   ├── train.py       # Training script
│   └── inference.py   # Inference script
├── sim/
│   ├── environment.py # DirectRLEnv for RL training
│   └── launch_app.py  # App launcher with flag handler
├── utils/
│   └── config.py      # Configuration loader
├── .gitignore
├── LICENSE
└── README.md
```

## Requirements

- IsaacLab + Isaac Sim installed (https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- CUDA GPU recommended (RTX 4060 or above)

## Installation
- Follow the installation steps for IsaacLab above
- Activate the IsaacLab environment (if used)

## Scripts
- The following will begin training with seed=0
```text
python scripts/train.py --seed=0
```
- The following will run a trained model with seed=1
```text
python scripts/inference.py --seed=1
```
- model/ contains all pre-trained models with the format "franka-[SEED].pth"
All raw configurations are stored at configs/yaml/

## Task Definition

Success Condition:
- Cuboid reaches 1 meter above the ground plane

Reward Components:
- Panda moves closer to cuboid
- Panda lifts cuboid
- Panda reaches reward condition
- Penalty on angular velocity
- Penalty on duration

Termination Conditions:
- Timeout: Episode exceeds 5 seconds (by default)
- Success: Success condition is outlined above