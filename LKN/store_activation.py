import torch 
import numpy as np 
from LKN.utils import get_llm_block

class KeyHiddenStateHook:
    def __init__(self):
        self.hidden_states = None
        self.probe_positions = None
        
    def __call__(self, module, input, output):
        # Store hidden states on CPU immediately and convert to float16
        self.hidden_states = []
        for i  in range(input[0].shape[0]):
            probe_position = self.probe_positions[0].to(input[0].device)
            self.hidden_states.append(input[0][i, probe_position, :].clone().detach().cpu().half())
        self.hidden_states = torch.stack(self.hidden_states)
        
    def clear(self):
        self.hidden_states = None


class CorrelationBasedNeuronAttribution:
    def __init__(self, model, tokenizer, model_name):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        
    def batchifed_tokenize(self, samples, batch_size):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        input_ids = [] 
        attention_mask = []
        num_batch = len(samples) // batch_size + 1 if len(samples) % batch_size != 0 else len(samples) // batch_size
        for i in range(num_batch):
            batch_samples = samples[i*batch_size:(i+1)*batch_size]
            inputs = self.tokenizer(batch_samples, return_tensors="pt", padding=True, truncation=True)
            input_ids.append(inputs.input_ids)
            attention_mask.append(inputs.attention_mask)
        return input_ids, attention_mask

    def store_activation(self, positive_samples, negative_samples, batch_size):
        blocks = get_llm_block(self.model, self.model_name)
        layer_indices = list(range(len(blocks)))
        hooks = []
        hook_handles = []
        for layer_idx in layer_indices:
            hook = KeyHiddenStateHook()
            # last six tokens
            hook.probe_positions = torch.tensor([[-6, -5, -4, -3, -2, -1]]).repeat(batch_size, 1)
            key_activation =  blocks[layer_idx].mlp.down_proj
            hook_handle = key_activation.register_forward_hook(hook)
            hooks.append(hook)
            hook_handles.append(hook_handle)
            
        means = {}
        variances = {}
        corrs = {}
        layer_key_hidden_states = {}
        for samples, label in zip([positive_samples, negative_samples], ["positive", "negative"]):
            means[label] = {}
            variances[label] = {}
            layer_key_hidden_states[label] = {layer: [] for layer in layer_indices}
            input_ids_all, attention_mask_all = self.batchifed_tokenize(samples, batch_size)
            
            for i in range(len(input_ids_all)):
                input_ids = input_ids_all[i]
                attention_mask = attention_mask_all[i]
                with torch.no_grad():
                    output = self.model(input_ids.to(self.model.device), 
                                        attention_mask=attention_mask.to(self.model.device))
                
                for index, sample in enumerate(samples):
                    for idx, layer_idx in enumerate(layer_indices):
                        key_hidden_states = hooks[idx].hidden_states  # Shape: [batch_size, sequence_length, hidden_size]
                        last_token_key_hidden_state = key_hidden_states[index,:,:]  # Shape: [hidden_size]
                        last_token_key_hidden_state = last_token_key_hidden_state.reshape(-1, last_token_key_hidden_state.shape[-1])
                        layer_key_hidden_states[label][layer_idx].append(last_token_key_hidden_state)

        for layer_idx in layer_indices:
            concat_input = []
            concat_target = []
            for label in ["positive", "negative"]:
                layer_key_hidden_states[label][layer_idx] = torch.concat(layer_key_hidden_states[label][layer_idx], dim=0)
                means[label][layer_idx] = layer_key_hidden_states[label][layer_idx].mean(dim=0).detach().cpu()
                variances[label][layer_idx] = layer_key_hidden_states[label][layer_idx].var(dim=0).detach().cpu()
                concat_input.append(layer_key_hidden_states[label][layer_idx])
                if label == "positive":
                    concat_target.append(torch.ones(len(layer_key_hidden_states[label][layer_idx]), 
                                                   device=layer_key_hidden_states[label][layer_idx].device))
                else:
                    concat_target.append(torch.zeros(len(layer_key_hidden_states[label][layer_idx]),
                                                    device=layer_key_hidden_states[label][layer_idx].device))
            concat_input = torch.concat(concat_input, dim=0)  # [n_samples, hidden_size]
            concat_target = torch.concat(concat_target, dim=0)  # [n_samples]
            
            # Vectorized correlation computation for all neurons at once
            # Center the data
            concat_input_centered = concat_input - concat_input.mean(dim=0, keepdim=True)  # [n_samples, hidden_size]
            concat_target_centered = concat_target - concat_target.mean()  # [n_samples]
            
            # Compute correlation: (x - x_mean) @ (y - y_mean) / (std(x) * std(y) * (n-1))
            numerator = (concat_input_centered * concat_target_centered[:, None]).sum(dim=0)  # [hidden_size]
            std_input = concat_input_centered.std(dim=0, unbiased=True)  # [hidden_size]
            std_target = concat_target_centered.std(unbiased=True)  # scalar
            denominator = std_input * std_target * (len(concat_target) - 1)
            
            # Compute correlations for all neurons at once
            correlations = numerator / denominator  # [hidden_size]
            
            # Convert to dictionary format for compatibility
            corrs[layer_idx] = {neuron_index: correlations[neuron_index]
                               for neuron_index in range(len(correlations))}
    
        for hook in hooks:
            hook.clear()
        
        self.means = means
        self.variances = variances
        self.corrs = corrs

