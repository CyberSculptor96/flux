cd /wangbenyou/huanghj/workspace/tmp/flux/prcv

export HF_ENDPOINT=https://hf-mirror.com

# flux 参数
num_steps=10
shift_flag=True     # option: False, True
method=euler         # option: euler, heun
seed=42
resolution=1024
mu=1.15  # mu is not used when shift=False
sigma=1.0
echo "[Config] num_steps=$num_steps, seed=$seed, shift=$shift_flag, method=$method, resolution=$resolution, mu=$mu, sigma=$sigma"

setting="shift_${shift_flag}_method_${method}_n_${num_steps}_mu_${mu}_resolution_${resolution}_seed_${seed}"
prompt_json_path="/wangbenyou/huanghj/workspace/tmp/flux/data/t2i-diversity-evalprompts/prompts_short_100_by_id.json"
image_path="/wangbenyou/huanghj/workspace/tmp/flux/output/prompts/$setting/"

cd clipscore && \
python clipscore.py \
    $prompt_json_path \
    $image_path \
    --setting $setting \
    --save_per_instance ../../results/clipscore_100.jsonl \
    2>&1 | tee ../../logs/evaluate/clipscore_100/$setting.log


# cd ../aesthetic-predictor && \
# python batch_inference.py \
#     --image_dir $image_path \
#     --setting $setting \
#     --save_per_instance ../../results/aesthetic_100.jsonl \
#     2>&1 | tee ../../logs/evaluate/aesthetic_100/$setting.log


# cd ../ImageReward && \
# python batch_inference.py \
#     --image_dir $image_path \
#     --prompt_dict $prompt_json_path \
#     --setting $setting \
#     --save_per_instance ../../results/imagereward_100.jsonl \
#     2>&1 | tee ../../logs/evaluate/imagereward_100/$setting.log