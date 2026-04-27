export CUDA_VISIBLE_DEVICES=7
source /mnt/T19/anaconda3/envs/Invad_env/bin/activate
PYTHONPATH=. python -m src.evaluate_heatmap \
    --eval_strategy inversion \
    --save_dir results/exp_dit_ad \
    --use_best_model \
    --eval_step 3