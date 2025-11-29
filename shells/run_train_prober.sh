
model_names=(
    'meta-llama/Meta-Llama-3.1-8B-Instruct'
    'Qwen/Qwen2.5-7B-Instruct'
)

batch_size=16
# only_last_token=True
init_seed=42
lr=1e-4
epochs=50
eval_every=10
only_last_token=False

for model_name in ${model_names[@]}; do
    python scripts/train_prober.py \
        --model_name ${model_name} \
        --batch_size ${batch_size} \
        --only_last_token ${only_last_token} \
        --init_seed ${init_seed} \
        --lr ${lr} \
        --epochs ${epochs} \
        --eval_every ${eval_every}
done