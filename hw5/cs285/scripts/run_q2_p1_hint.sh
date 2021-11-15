#!/bin/bash

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \
    --exp_name q2_dqn_shift2_scale100 \
    --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0 \
    --exploit_rew_shift 2 --exploit_rew_scale 100

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \
    --exp_name q2_cql_shift2_scale100 \
    --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 \
    --exploit_rew_shift 2 --exploit_rew_scale 100