import torch 
from LKN.utils import get_llm_block

class KeyHiddenStateHook:
    def __init__(self):
        self.hidden_states = None
        self.probe_positions = None
        
    def __call__(self, module, input, output):
        # Store hidden states on CPU immediately and convert to float16
        self.hidden_states = []
        for i  in range(self.probe_positions.shape[0]):
            probe_position = self.probe_positions[i]
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
        for sample in samples:
            inputs = self.tokenizer(sample, return_tensors="pt", padding=True, truncation=True)
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
            input_ids, attention_mask = self.batchifed_tokenize(samples, batch_size)
            with torch.no_grad():
                output = self.model(input_ids, attention_mask=attention_mask)
            
            for index, sample in enumerate(samples):
                sample_id = sample_id.item()
                for idx, layer_idx in enumerate(layer_indices):
                    key_hidden_states = hooks[idx].hidden_states  # Shape: [batch_size, sequence_length, hidden_size]
                    last_token_key_hidden_state = key_hidden_states[index,:,:]  # Shape: [hidden_size]
                    layer_key_hidden_states[label][layer_idx].append(last_token_key_hidden_state)

        for layer_idx in layer_indices:
            layer_key_hidden_states[label][layer_idx] = torch.stack(layer_key_hidden_states[label][layer_idx])
            means[label][layer_idx] = layer_key_hidden_states[label][layer_idx].mean(dim=0).detach().cpu().half()
            variances[label][layer_idx] = layer_key_hidden_states[label][layer_idx].var(dim=0).detach().cpu().half()
            concat_input = torch.cat([means[label][layer_idx], variances[label][layer_idx]], dim=0)
            concat_target = torch.cat([torch.ones(len(means[label][layer_idx])), torch.zeros(len(variances[label][layer_idx]))], dim=0)
            corrs[label][layer_idx] = torch.corrcoef(concat_input, concat_target)
        
        for hook in hooks:
            hook.clear()
        
        return means, variances