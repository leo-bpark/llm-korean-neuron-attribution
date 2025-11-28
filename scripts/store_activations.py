import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch 
from tqdm import tqdm
from LKN.prompts import formatting_prompts
from LKN.utils import get_model, get_llm_block, get_mlp_down_proj
from LKN.utils import get_dataset

def main(args, model, tokenizer):
    args.save_dir = os.path.join(
        f'outputs/activations/',
        f'{args.model_name}/{args.concept}/{args.split}',
    )
    os.makedirs(args.save_dir, exist_ok=True)
    # ------------------------------------------------------------------------------------
    # Hooks 
    # ------------------------------------------------------------------------------------
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

    class ValueHiddenStateHook:
        def __init__(self):
            self.hidden_states = None
            
        def __call__(self, module, input, output):
            # Store hidden states on CPU immediately and convert to float16
            self.hidden_states = []
            for i in range(self.probe_positions.shape[0]):
                probe_position = self.probe_positions[i]
                self.hidden_states.append(output[i, probe_position, :].clone().detach().cpu().half())
            self.hidden_states = torch.stack(self.hidden_states)
            
        def clear(self):
            self.hidden_states = None

    key_hooks = []
    value_hooks = []
    key_hook_handles = []
    value_hook_handles = []
    
    blocks = get_llm_block(model, args.model_name)
    args.layer_indices = list(range(len(blocks)))  # [::4] + [len(blocks) - 1] # every 4 layers and the last layer
    # choose the middle layers 
    # center = len(args.layer_indices)//2
    # args.layer_indices = args.layer_indices[center-1:center+1]
    
    for layer_idx in args.layer_indices:
        key_hook = KeyHiddenStateHook()
        key_activation = get_mlp_down_proj(args.model_name, blocks[layer_idx])
        handle_key = key_activation.register_forward_hook(key_hook)
        key_hooks.append(key_hook)
        key_hook_handles.append(handle_key)
        
        value_hook = ValueHiddenStateHook()
        value_activation = get_mlp_down_proj(args.model_name, blocks[layer_idx])
        handle_value = value_activation.register_forward_hook(value_hook)
        value_hooks.append(value_hook)
        value_hook_handles.append(handle_value)

    # ------------------------------------------------------------------------------------
    # Main Evaluation 
    # ------------------------------------------------------------------------------------
    layer_key_hidden_states = {layer: [] for layer in args.layer_indices}
    layer_value_hidden_states = {layer: [] for layer in args.layer_indices}
    all_labels = []
    dataset = get_dataset(args.concept, args.split, data_dir="data/sample.json")
    dataloader = formatting_prompts(args.model_name, dataset, tokenizer, args.batch_size)
    pbar = tqdm(dataloader, desc=f"[Save Activation MLP] {args.save_dir}")
    print("--------------------------------")
    print(len(dataset))
    print(len(dataloader))
    print("--------------------------------")
    
    try:
        for batch in pbar:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            sample_ids = batch['sample_ids']
            probe_positions = batch['probe_positions']
            labels = batch['labels']
            for key_hook in key_hooks:
                key_hook.probe_positions = probe_positions
            for value_hook in value_hooks:
                value_hook.probe_positions = probe_positions
            
            with torch.no_grad():
                generated_ids_batch = model.forward(input_ids, 
                                                    attention_mask=attention_mask,)  
                torch.cuda.empty_cache()  
                input_ids = input_ids.cpu()
                attention_mask = attention_mask.cpu()
                
            labels = labels.unsqueeze(1)
            expanded_labels = labels.repeat_interleave(probe_positions.shape[1]).reshape(labels.shape[0], -1)
            all_labels.append(expanded_labels)
            for index, sample_id in enumerate(sample_ids):
                sample_id = sample_id.item()
                for idx, layer_idx in enumerate(args.layer_indices):
                    key_hidden_states = key_hooks[idx].hidden_states  # Shape: [batch_size, sequence_length, hidden_size]
                    value_hidden_states = value_hooks[idx].hidden_states  # Shape: [batch_size, sequence_length, hidden_size]
                    last_token_key_hidden_state = key_hidden_states[index,:,:]  # Shape: [hidden_size]
                    last_token_value_hidden_state = value_hidden_states[index,:,:]  # Shape: [hidden_size]
                    layer_key_hidden_states[layer_idx].append(last_token_key_hidden_state)
                    layer_value_hidden_states[layer_idx].append(last_token_value_hidden_state)
                    
            # Clear hooks after processing each batch
            for key_hook in key_hooks:
                key_hook.clear()
            for value_hook in value_hooks:
                value_hook.clear()
                
    finally:
        # Ensure hooks are removed even if an error occurs
        for handle in key_hook_handles:
            handle.remove()
        for handle in value_hook_handles:
            handle.remove()
            
        # Save the accumulated hidden states
        all_labels = torch.concat(all_labels, dim=0)
        print("Labels", all_labels.shape)
        torch.save(all_labels, os.path.join(args.save_dir, f"labels.pt"))
        
        for layer_idx in args.layer_indices:
            key_hidden_states = torch.stack(layer_key_hidden_states[layer_idx], dim=0)
            value_hidden_states = torch.stack(layer_value_hidden_states[layer_idx], dim=0)
            torch.save(key_hidden_states, os.path.join(args.save_dir, f"layer_{layer_idx}_key.pt"))
            torch.save(value_hidden_states, os.path.join(args.save_dir, f"layer_{layer_idx}_value.pt"))
            
            # Clear the layer's hidden states after saving
            layer_key_hidden_states[layer_idx] = []
            layer_value_hidden_states[layer_idx] = []
            print("Key", key_hidden_states.shape, "Value", value_hidden_states.shape)
            
        # Final cleanup
        torch.cuda.empty_cache()

# ------------------------------------------------------------------------------------
# Main 
# ------------------------------------------------------------------------------------
if __name__ == "__main__":  
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    import json 
    model, tokenizer = get_model(args.model_name)
    dataset = json.load(open("data/sample.json", "r"))
    for concept in dataset.keys():
        for split in ["train", "test"]:
            args.concept = concept
            args.split = split
            main(args, model, tokenizer)