num_steps=10
shift=True      # option: False, True
mu=0.3          # only used when shift=True
method=heun     # option: euler, heun
seed=42
echo "num_steps: $num_steps, seed: $seed, shift: $shift, mu: $mu, method: $method"

output_dir=output/seed_$seed/n_${num_steps}_$method/
mkdir -p $output_dir

python -m flux t2i --name flux-dev \
  --height 1024 --width 1024 \
  --prompt "a cat sitting on a bench in tokyo" \
  --num_steps $num_steps \
  --mu $mu \
  --seed $seed \
  --output_dir $output_dir \
  --shift $shift \
  --method $method \
  2>&1 | tee logs/inference_n_$num_steps.log \
#   --seed 42 \
