def get_llm_block(llm, llm_name):
    if llm_name == "gpt2":
        block = llm.transformer.h
    elif 'meta-llama' in llm_name:
        block = llm.model.layers
    elif 'Qwen' in llm_name:
        block = llm.model.layers
    else:
        raise ValueError(f"Unsupported model: {llm_name}")
    return block



from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_model(name:str, device_map="auto"):
    """
    Get model and tokenizer
    Args:
        name: model name
    Returns:
        model and tokenizer
    """
    # For decoder-only models, left padding is crucial for correct generation
    model = AutoModelForCausalLM.from_pretrained(name, device_map=device_map, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(name, device_map=device_map, padding_side="left")
    assert tokenizer.chat_template is not None, "Tokenizer does not have a chat template"
    # Ensure consistent token settings
    tokenizer.pad_token = tokenizer.eos_token
    
    # Double-check padding side is set correctly
    if tokenizer.padding_side != "left":
        print(f"Warning: Tokenizer padding_side was not set to 'left'. Setting it now.")
        tokenizer.padding_side = "left"
    
    return model, tokenizer



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
