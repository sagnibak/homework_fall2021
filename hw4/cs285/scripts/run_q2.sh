#!/bin/bash

python cs285/scripts/run_hw4_mb.py --exp_name \
    hw4_mb_q2_obstacles_singleiteration --env_name obstacles-cs285-v0 \
    --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 \
    --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 \
    --seed 1 \
    --mpc_action_sampling_strategy 'random'

# python cs285/scripts/run_hw4_mb.py --exp_name \
#     hw4_mb_q2_obstacles_singleiteration --env_name obstacles-cs285-v0 \
#     --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 \
#     --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 \
#     --seed 3829 # \
#     --mpc_action_sampling_strategy 'random' \

# python cs285/scripts/run_hw4_mb.py --exp_name \
#     hw4_mb_q2_obstacles_singleiteration --env_name obstacles-cs285-v0 \
#     --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 \
#     --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 \
#     --seed 101 # \
#     --mpc_action_sampling_strategy 'random' \

# python cs285/scripts/run_hw4_mb.py --exp_name \
#     hw4_mb_q2_obstacles_singleiteration --env_name obstacles-cs285-v0 \
#     --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 \
#     --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 \
#     --seed 12231 # \
#     --mpc_action_sampling_strategy 'random' \

# python cs285/scripts/run_hw4_mb.py --exp_name \
#     hw4_mb_q2_obstacles_singleiteration --env_name obstacles-cs285-v0 \
#     --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 \
#     --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 \
#     --seed 392873 # \
#     --mpc_action_sampling_strategy 'random' \

# python cs285/scripts/run_hw4_mb.py --exp_name \
#     hw4_mb_q2_obstacles_singleiteration --env_name obstacles-cs285-v0 \
#     --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 \
#     --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 \
#     --seed 44232 # \
#     --mpc_action_sampling_strategy 'random' \

# python -m pdb -c continue # for debugging