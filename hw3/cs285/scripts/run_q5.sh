python cs285/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v2 --ep_len 1000 \
    --discount 0.95 -n 200 -l 2 -s 64 -b 5000 -lr 0.01 \
    --exp_name hw3_q5_20_20 -ntu 20 -ngsptu 20

python cs285/scripts/run_hw3_actor_critic.py --env_name HalfCheetah-v2 --ep_len 150 \
    --discount 0.90 --scalar_log_freq 1 -n 200 -l 2 -s 32 -b 30000 -eb 1500 \
    -lr 0.02 --exp_name hw3_q5_20_20 -ntu 20 -ngsptu 20