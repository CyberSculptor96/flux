#!/bin/bash
set -e

num_steps=10
shift_flag=True     # option: False, True
method=heun         # option: euler, heun
seed=42
prompt="a cat sitting on a bench in tokyo"

# 8 个 mu 的取值（你可以按实验需求修改）
mus=(0.3 0.5 0.6 1.0 1.15 1.4 1.8 2.0)

echo "[Config] num_steps=$num_steps, seed=$seed, shift=$shift_flag, method=$method"
echo "Running 8 parallel experiments..."

for i in ${!mus[@]}; do
    mu=${mus[$i]}
    gpu_id=$i

    output_dir="output/seed_${seed}/n_${num_steps}_${method}/"
    mkdir -p "$output_dir"

    echo "Launching GPU $gpu_id with mu=$mu  →  $output_dir"

    CUDA_VISIBLE_DEVICES=$gpu_id \
    python -m flux t2i \
        --name flux-dev \
        --height 1024 \
        --width 1024 \
        --prompt "$prompt" \
        --num_steps $num_steps \
        --mu $mu \
        --seed $seed \
        --output_dir $output_dir \
        --shift $shift_flag \
        --method $method \
        > logs/inference_n_${num_steps}_mu_${mu}.log 2>&1 &

done

wait
echo "✅ All 8 experiments finished."
