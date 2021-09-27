import os


# BATCHES = [5, 10, 50, 100, 200, 600, 1000, 2000]
# BATCHES = [10000, 40000]
BATCHES = [32, 128, 512, 1024]
LEARNING_RATES = [lr for i in range(-4, -1) for lr in (10 ** i, 4 * (10 ** i))]

# TEMPLATE = """python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 \
#     --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b {batch} -lr {lr} -rtg \
#     --exp_name q2_b{batch}_r{lr}
# """

TEMPLATE = """python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 \
    --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 128 -b {batch} -lr {lr} -rtg \
    --exp_name q2_b{batch}_r{lr}_s128
"""

for b in BATCHES:
    for lr in LEARNING_RATES:
        if os.system(TEMPLATE.format(batch=b, lr=lr)) != 0:
            raise RuntimeError("Got nonzero exit code.")