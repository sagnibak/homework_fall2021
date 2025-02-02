#!/bin/bash

python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --use_rnd \
    --unsupervised_exploration --exp_name q1_env2_rnd
python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 \
    --unsupervised_exploration --exp_name q1_env2_random

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
    --unsupervised_exploration --exp_name q1_env1_rnd
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \
    --unsupervised_exploration --exp_name q1_env1_random