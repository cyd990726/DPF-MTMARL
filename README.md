# DPF-MTMARL

## Code framework
DPF-MTMARL: Dual-Policy Fusion for Multi-Task Multi-Agent Reinforcement Learning

This repository is based on the PyMARL (https://github.com/oxwhirl/pymarl). For the installation instructions, pls refer to PyMARL.

## Key code
The codes related to DPF-MTMARL are,
1. Algorithm config: /src/config/algs/xdistral_train_beta.yaml
2. Multi-task config: /src/config/tasks/
3. Learners: /src/learners/multi-task/xdistral_learner.py
4. Controllers: /src/Controllers/multi-task/xdistral_controller.py

## Run the code
To run experiment, for example stalkers_and_zealots, pls use the following command:

`nohup python3 -u src/main.py --config=xdistral_train --task-config=stalkers_and_zealots --use_tensorboard=True --remark=1 &> 1.out &`

