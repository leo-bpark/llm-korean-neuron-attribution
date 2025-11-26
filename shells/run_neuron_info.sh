
model_names=(
    'meta-llama/Meta-Llama-3.1-8B-Instruct'
    'Qwen/Qwen2.5-7B-Instruct'
)

batch_size=16
for model_name in ${model_names[@]}; do
    python scripts/run_neuron_stats.py --model_name ${model_name} --batch_size ${batch_size}
done