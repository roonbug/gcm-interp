export RM_INTERP_REPO="/home/ubuntu/gcm-interp/"
echo "RM_INTERP_REPO is $RM_INTERP_REPO"

declare -a pairs=(
  "harmful_harmless"
)
declare -A eval_datasets

algos=("acp" "probes" "random" "atp" "atp-zero")
model_id="upstage/SOLAR-10.7B-Instruct-v1.0"
judge_id="meta-llama/Llama-3.1-70B-Instruct"
device="cuda:0"

for pair in "${pairs[@]}"; do
  IFS='_' read -r source base <<< "$pair"
  for algo in "${algos[@]}"; do
      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source "$source" \
                    --base "$base" \
                    --judge_id "$judge_id" \
                    --device "$device" \
                    --eval_model \
                    --eval_test \
                    --N "0" \
                    --pyreft
  done
done
