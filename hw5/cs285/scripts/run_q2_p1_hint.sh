#!/bin/bash

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \
    --exp_name q2_dqn_shift-1_scale10 \
    --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0 \
    --exploit_rew_shift -1 --exploit_rew_scale 10

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \
    --exp_name q2_cql_shift-1_scale10 \
    --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 \
    --exploit_rew_shift -1 --exploit_rew_scale 10