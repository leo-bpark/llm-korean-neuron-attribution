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


def attribute(llm, tokenizer, model_name, input_text, neurons):
    tokens = tokenizer.encode(input_text, return_tensors='pt').to(llm.device)  
    embed_layer = llm.get_input_embeddings()           # nn.Embedding
    embeds = embed_layer(tokens)                       # [1, T, d]
    embeds.retain_grad()                               # 이 텐서에 grad 저장

    blocks = get_llm_block(llm, model_name)
    hooks = [BackpropHook() for _ in neurons]
    for hook, (target_layer, target_neuron) in zip(hooks, neurons):
        hook.probe_positions = torch.tensor([[ -2, -1]])  # batch=1 기준
        layer_module = blocks[target_layer]
        layer_module.register_forward_hook(hook)

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
    return result