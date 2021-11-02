#!/bin/bash

python cs285/scripts/run_hw4_mb.py --exp_name q3_obstacles2 \
    --env_name obstacles-cs285-v0 --add_sl_noise \
    --num_agent_train_steps_per_iter 50 --batch_size_initial 5000 \
    --batch_size 1000 --mpc_horizon 10 \
    --n_iter 30 --mpc_action_sampling_strategy 'random'

python cs285/scripts/run_hw4_mb.py --exp_name q3_reacher2 \
    --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 10 \
    --num_agent_train_steps_per_iter 1000 --batch_size_initial 5000 \
    --batch_size 5000 --n_iter 50 --mpc_action_sampling_strategy 'random'

python cs285/scripts/run_hw4_mb.py --exp_name q3_cheetah2 \
    --env_name cheetah-cs285-v0 --mpc_horizon 15 --add_sl_noise \
    --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 \
    --batch_size 5000 --n_iter 60 --mpc_action_sampling_strategy 'random'