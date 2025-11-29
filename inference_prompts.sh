#!/bin/bash
set -e

export HF_ENDPOINT=https://hf-mirror.com

JSON_FILE="/wangbenyou/huanghj/workspace/tmp/flux/data/t2i-diversity-evalprompts/prompts_short_100_by_id.json"

# flux 参数
num_steps=10
shift_flag=True      # option: False, True
method=euler         # option: euler, heun
seed=42
model_name=flux-dev
resolution=1024
mu=1.15  # mu is not used when shift=False
sigma=2.0
echo "[Config] num_steps=$num_steps, seed=$seed, shift=$shift_flag, method=$method, resolution=$resolution, mu=$mu, sigma=$sigma"

# 每张卡的数量
NUM_GPUS=8

echo "[INFO] Loading prompts from $JSON_FILE ..."
# 读取所有 prompt_id
prompt_ids=($(jq -r 'keys[]' $JSON_FILE))
total=${#prompt_ids[@]}
echo "[INFO] Total prompts: $total"

# 计算每卡均分数
per_gpu=$(( (total + NUM_GPUS - 1) / NUM_GPUS ))
echo "[INFO] Each GPU will process ~$per_gpu prompts."

sleep 5

# 遍历 GPU
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do

    (
        start=$(( gpu * per_gpu ))
        end=$(( start + per_gpu ))

        echo "[GPU $gpu] processing prompts index $start to $end"

        for ((i=start; i<end && i<total; i++)); do
            pid=${prompt_ids[$i]}
            prompt=$(jq -r --arg k "$pid" '.[$k]' $JSON_FILE)

            echo "[GPU $gpu] prompt_id=$pid"

            outdir="output/prompts/shift_${shift_flag}_method_${method}_n_${num_steps}_mu_${mu}_sigma_${sigma}_resolution_${resolution}_seed_${seed}/"
            mkdir -p "$outdir"

            CUDA_VISIBLE_DEVICES=$gpu \
            python -m flux t2i \
                --name $model_name \
                --height $resolution \
                --width $resolution \
                --prompt "$prompt" \
                --num_steps $num_steps \
                --seed $seed \
                --output_dir "$outdir" \
                --shift $shift_flag \
                --mu $mu \
                --sigma $sigma \
                --method $method \
                --prompt_id $pid \
                2>&1 | tee "logs/inference/inference_${pid}-method_${method}-mu_${mu}-n_${num_steps}-gpu_${gpu}.log"

        done
    ) &

done

wait
echo "🎉 All prompts processed on $NUM_GPUS GPUs."
