import os
import sys

b_star = 30000
r_star = 0.02

COMMAND = """python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
    --discount 0.95 -n 100 -l 2 -s 32 -b {b_star} -lr {r_star} \
    --exp_name q4_b{b_star}_r{r_star}

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
    --discount 0.95 -n 100 -l 2 -s 32 -b {b_star} -lr {r_star} -rtg \
    --exp_name q4_b{b_star}_r{r_star}_rtg

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
    --discount 0.95 -n 100 -l 2 -s 32 -b {b_star} -lr {r_star} --nn_baseline \
    --exp_name q4_b{b_star}_r{r_star}_nnbaseline

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
    --discount 0.95 -n 100 -l 2 -s 32 -b {b_star} -lr {r_star} -rtg --nn_baseline \
    --exp_name q4_b{b_star}_r{r_star}_rtg_nnbaseline""".format(
        b_star=b_star,
        r_star=r_star
    )


sys.exit(os.system(COMMAND))
