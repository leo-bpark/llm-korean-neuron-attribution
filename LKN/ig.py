import torch 
from LKN.utils import  get_llm_block

class BackpropHook:
        def __init__(self):
            self.hidden_states = None
            self.probe_positions = None
            
        def __call__(self, module, input, output):
            # Store hidden states on CPU immediately and convert to float16
            self.hidden_states = []
            for i  in range(self.probe_positions.shape[0]):
                probe_position = self.probe_positions[i]
                self.hidden_states.append(input[0][i, probe_position, :])
            self.hidden_states = torch.stack(self.hidden_states)
            
        def clear(self):
            self.hidden_states = None


def attribute(llm, tokenizer, model_name, input_text, neurons, probe_positions=None):
    tokens = tokenizer.encode(input_text, return_tensors='pt').to(llm.device)  
    embed_layer = llm.get_input_embeddings()           # nn.Embedding
    embeds = embed_layer(tokens)                       # [1, T, d]
    embeds.retain_grad()                               # 이 텐서에 grad 저장

    if probe_positions is None:
        probe_positions =  [i for i in range(len(tokens[0]))]
        
    blocks = get_llm_block(llm, model_name)
    hooks = [BackpropHook() for _ in neurons]
    hook_handles = []
    for hook, (target_layer, target_neuron) in zip(hooks, neurons):
        hook.probe_positions = torch.tensor([probe_positions]).to(llm.device)  # batch=1 기준
        layer_module = blocks[target_layer].mlp.down_proj
        hook_handle = layer_module.register_forward_hook(hook)
        hook_handles.append(hook_handle)
    llm.zero_grad()
    output = llm(inputs_embeds=embeds)

    # ③ 타겟 뉴런에서 backward
    device = embeds.device
    sums = []
    for hook, (target_layer, target_neuron) in zip(hooks, neurons):
        # hidden_states를 같은 device로 이동하고 target_neuron 인덱싱
        hidden = hook.hidden_states.to(device)
        sums.append(hidden[:, :, target_neuron].sum())
    sum_of_sums = torch.sum(torch.stack(sums))
    sum_of_sums.backward()
    input_grad = embeds.grad * embeds
    batch_attribution = input_grad.abs().mean(dim=-1)
    batch_index = 0 
    result = []
    # 토큰 ID를 직접 사용하여 더 안전한 디코딩 가능
    token_ids = tokens[batch_index].cpu().tolist()
    for token_id, attr in zip(token_ids, batch_attribution[batch_index]):
        # 토큰 ID를 튜플로 저장 (token_id, attr)
        # utils.py에서 token_id를 직접 디코딩할 수 있도록
        result.append((token_id, attr))
        
    # clean up
    for handle in hook_handles:
        handle.remove()
    return result

def activation_recording(llm, tokenizer, model_name, input_text, neurons, probe_positions=None):
    tokens = tokenizer.encode(input_text, return_tensors='pt').to(llm.device)  
    embed_layer = llm.get_input_embeddings()           # nn.Embedding
    embeds = embed_layer(tokens)                       # [1, T, d]

    if probe_positions is None:
        probe_positions =  [i for i in range(len(tokens[0]))]
        
    blocks = get_llm_block(llm, model_name)
    hooks = [BackpropHook() for _ in neurons]
    hook_handles = []
    for hook, (target_layer, target_neuron) in zip(hooks, neurons):
        hook.probe_positions = torch.tensor([probe_positions]).to(llm.device)  # batch=1 기준
        layer_module = blocks[target_layer].mlp.down_proj
        hook_handle = layer_module.register_forward_hook(hook)
        hook_handles.append(hook_handle)
    llm.zero_grad()
    output = llm(inputs_embeds=embeds)
    
    # 각 뉴런의 활성화 값을 추출하고 평균내기
    device = embeds.device
    neuron_activations = []
    for hook, (target_layer, target_neuron) in zip(hooks, neurons):
        # hidden_states: [batch, len(probe_positions), hidden_dim] (batch=1)
        # target_neuron 인덱스의 활성화 값 추출: [len(probe_positions)]
        hidden = hook.hidden_states.to(device)
        # attribute 함수와 동일한 인덱싱 사용
        neuron_activations.append(hidden[:, :, target_neuron])  # [len(probe_positions)]

    
    batch_index = 0
    # attribute 함수와 같은 형태로 반환: [(token_id, activation), ...]
    averaged_activations = torch.stack(neuron_activations).mean(dim=0)[batch_index]
    token_ids = tokens[batch_index].cpu().tolist()
    result = []
    for token_id, activation in zip(token_ids, averaged_activations):
        result.append((token_id, activation))
    
    # clean up
    for handle in hook_handles:
        handle.remove()
    
    return result