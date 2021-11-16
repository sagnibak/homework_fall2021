#!/bin/bash

for alpha in 0.02 0.5
do
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \
    --use_rnd --unsupervised_exploration --offline_exploitation \
    --cql_alpha=${alpha} --exp_name q2_alpha${alpha}
done