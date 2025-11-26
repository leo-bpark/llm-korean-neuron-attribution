

from LKN.store_activation import CorrelationBasedNeuronAttribution
from LKN.utils import get_model, format_chat_template

import os 
import json
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()

model_name = args.model_name
batch_size = args.batch_size

model, tokenizer = get_model(model_name)
json_file = open("data/sample.json", "r")
data = json.load(json_file)

results = {}

save_dir = f"outputs/model_info/{model_name}"
os.makedirs(save_dir, exist_ok=True)

for concept in tqdm(data.keys()):
    positive_samples = [sample for sample in data[concept]["pos"]]
    negative_samples = [sample for sample in data[concept]["neg"]]
    
    positive_samples = [format_chat_template(tokenizer, model_name, sample) for sample in positive_samples]
    negative_samples = [format_chat_template(tokenizer, model_name, sample) for sample in negative_samples]
 
    atr = CorrelationBasedNeuronAttribution(model, tokenizer, model_name)
    atr.store_activation(positive_samples, negative_samples, batch_size)
    
    num_layers = len(atr.means["positive"])
    num_neurons = atr.means["positive"][0].shape[0]
    print("num_layers", num_layers)
    print("num_neurons", num_neurons)

    # Convert to layer/neuron tensor format
    pos_mean_tensor = np.zeros((num_layers, num_neurons))
    pos_var_tensor = np.zeros((num_layers, num_neurons))
    neg_mean_tensor = np.zeros((num_layers, num_neurons))
    neg_var_tensor = np.zeros((num_layers, num_neurons))
    corr_tensor = np.zeros((num_layers, num_neurons))
    
    for layer_idx in range(num_layers):
        for neuron_index in range(num_neurons):
            pos_mean_tensor[layer_idx, neuron_index] = atr.means["positive"][layer_idx][neuron_index].item()
            pos_var_tensor[layer_idx, neuron_index] = atr.variances["positive"][layer_idx][neuron_index].item()
            neg_mean_tensor[layer_idx, neuron_index] = atr.means["negative"][layer_idx][neuron_index].item()
            neg_var_tensor[layer_idx, neuron_index] = atr.variances["negative"][layer_idx][neuron_index].item()
            corr_tensor[layer_idx, neuron_index] = atr.corrs[layer_idx][neuron_index].item()
    
    results[concept] = {
        'pos_mean': pos_mean_tensor.tolist(),
        'pos_var': pos_var_tensor.tolist(),
        'neg_mean': neg_mean_tensor.tolist(),
        'neg_var': neg_var_tensor.tolist(),
        'corr': corr_tensor.tolist()
    }

with open(f"{save_dir}/neuron_stats.json", "w") as f:
    json.dump(results, f)