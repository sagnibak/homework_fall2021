#!/bin/bash

for num_steps in 1000 5000 15000
do
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
        --num_exploration_steps=${num_steps} --offline_exploitation \
        --cql_alpha=0.1 --unsupervised_exploration \
        --exp_name q2_cql_numsteps_${num_steps}

    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
        --num_exploration_steps=${num_steps} --offline_exploitation \
        --cql_alpha=0.0 --unsupervised_exploration \
        --exp_name q2_cql_numsteps_${num_steps}
done