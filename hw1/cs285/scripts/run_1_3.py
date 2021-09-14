import os

TEMPLATE = \
"""python cs285/scripts/run_hw1.py \\
    --ep_len 1000 \\
    --eval_batch_size 5000 \\
    --expert_policy_file cs285/policies/experts/Ant.pkl \\
    --env_name Ant-v2 --exp_name bc_ant_{train_batch_size} --n_iter 1 \\
    --expert_data cs285/expert_data/expert_data_Ant-v2.pkl \\
    --n_layers 2 --size 128 \\
    --train_batch_size {train_batch_size} \\
    --video_log_freq -1"""

BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024]

for train_batch_size in BATCH_SIZES:
    os.system(TEMPLATE.format(train_batch_size=train_batch_size))