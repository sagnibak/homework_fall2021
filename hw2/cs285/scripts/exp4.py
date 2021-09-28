import os

BATCHES = [10_000, 30_000, 50_000]
LEARNING_RATES = [0.005, 0.01, 0.02]
TEMPLATE = """python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
    --discount 0.95 -n 100 -l 2 -s 32 -b {batch} -lr {lr} -rtg --nn_baseline \
    --exp_name q4_search_b{batch}_lr{lr}_rtg_nnbaseline"""

for b in BATCHES:
    for lr in LEARNING_RATES:
        if os.system(TEMPLATE.format(batch=b, lr=lr)) != 0:
            raise RuntimeError("Got nonzero exit code.")