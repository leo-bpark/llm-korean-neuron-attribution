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
import datasets

def get_dataset(concept, split, data_dir='/data/sample.json', seed = 42, proportion = 0.3):
    data = json.load(open(data_dir, "r"))
    concept_data = data[concept]
    num_samples = min(len(concept_data["pos"]), len(concept_data["neg"]))
    
    state = np.random.get_state()
    np.random.seed(seed)
    indices = np.random.permutation(num_samples)[:int(num_samples * proportion)]
    if split == "train":
        pass # use all samples
    else:
        indices = np.setdiff1d(np.arange(num_samples), indices)
    np.random.set_state(state)
    
    text = [] 
    label = []
    for index in indices:
        text.append(concept_data["pos"][index])
        label.append(1)
        text.append(concept_data["neg"][index])
        label.append(0)
    # text, label
    dataset = datasets.Dataset.from_list([{
        "text": text[i],
        "label": label[i],
        "sample_id": indices[i],
    } for i in range(len(indices))])
        
    return dataset


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



def format_chat_template(tokenizer, model_name, text):
    if 'meta-llama' in model_name:
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}, 
                    {"role": "user", "content": text}]
    elif 'Qwen' in model_name:
        messages = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}, 
                    {"role": "user", "content": text}]
    chat_formatted_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        return_tensors=None
    )
    return chat_formatted_text



import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import HTML, display
import numpy as np
import matplotlib.font_manager as fm
import html

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
# 한글을 위한 폰트 설정 시도
try:
    # 시스템에 있는 한글 폰트 찾기
    font_list = [f.name for f in fm.fontManager.ttflist]
    korean_fonts = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Noto Sans CJK KR', 'NanumBarunGothic']
    for font_name in korean_fonts:
        if font_name in font_list:
            plt.rcParams['font.family'] = font_name
            break
except:
    pass

def decode_and_merge_tokens(tokens, attributions, tokenizer):
    """Decode token IDs/strings and merge broken Korean pieces.

    이 함수는 notebook 시각화와 동일한 토큰 병합 로직을 사용합니다.
    프론트엔드 / 서버 양쪽에서 재사용할 수 있도록 분리했습니다.
    """
    # 원본 텍스트 복원: 모든 토큰 ID를 한번에 디코딩
    token_ids = []
    for token in tokens:
        if isinstance(token, int):
            token_ids.append(token)
        elif isinstance(token, str):
            try:
                if hasattr(tokenizer, 'convert_tokens_to_ids'):
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    token_ids.append(token_id)
                else:
                    # 토큰 ID를 얻을 수 없는 경우 스킵
                    continue
            except Exception:
                continue

    # 원본 텍스트 복원
    if token_ids:
        full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    else:
        full_text = ""

    # 개별 토큰들을 디코딩하고 원본 텍스트와 매칭
    decoded_tokens = []
    text_pos = 0  # 원본 텍스트에서의 현재 위치

    for i, token in enumerate(tokens):
        decoded = None
        # 토큰이 정수인 경우 (토큰 ID) 직접 디코딩
        if isinstance(token, int):
            try:
                decoded = tokenizer.decode([token], skip_special_tokens=False)
            except Exception:
                decoded = str(token)
        # 토큰이 문자열인 경우
        elif isinstance(token, str):
            try:
                if hasattr(tokenizer, 'convert_tokens_to_ids'):
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    decoded = tokenizer.decode([token_id], skip_special_tokens=False)
                else:
                    decoded = token
            except (KeyError, ValueError, TypeError):
                decoded = token
        elif isinstance(token, bytes):
            decoded = token.decode('utf-8', errors='replace')
        else:
            decoded = str(token)

        # 특수 토큰 처리
        if decoded:
            decoded = decoded.replace('Ġ', ' ').replace('▁', ' ')

        decoded_tokens.append(decoded if decoded else "")

    # 한국어 토큰 병합: 두 개로 나뉜 토큰을 하나로 합치기
    # 원본 텍스트를 기준으로 병합
    merged_tokens = []
    merged_attributions = []
    i = 0
    text_pos = 0

    while i < len(decoded_tokens):
        current_token = decoded_tokens[i]
        current_attr = attributions[i]

        # 다음 토큰과 합쳐서 원본 텍스트와 매칭 시도
        should_merge = False
        merged_text = current_token

        if i + 1 < len(decoded_tokens):
            next_token = decoded_tokens[i + 1]
            # 두 토큰의 원본 토큰 ID를 합쳐서 디코딩
            try:
                if isinstance(tokens[i], int) and isinstance(tokens[i + 1], int):
                    # 두 토큰을 합쳐서 디코딩
                    combined_decoded = tokenizer.decode(
                        [tokens[i], tokens[i + 1]], skip_special_tokens=False
                    )
                    combined_decoded = combined_decoded.replace("Ġ", " ").replace("▁", " ")

                    # 원본 텍스트에서 현재 위치 확인
                    if text_pos < len(full_text):
                        # 원본 텍스트에서 현재 위치부터 확인
                        remaining_text = full_text[text_pos:]

                        # 합친 결과가 원본 텍스트의 시작 부분과 일치하는지 확인
                        # 또는 깨진 문자가 복원되는 경우
                        has_broken_char = "" in current_token or "" in next_token
                        combined_has_broken = "" in combined_decoded

                        # 깨진 문자가 있고 합친 결과가 깨지지 않은 경우
                        if has_broken_char and not combined_has_broken:
                            should_merge = True
                            merged_text = combined_decoded
                        # 또는 합친 결과가 원본 텍스트의 시작 부분과 일치하는 경우
                        elif remaining_text.startswith(combined_decoded):
                            # 개별 토큰들이 원본 텍스트와 일치하지 않는 경우 병합
                            if not remaining_text.startswith(current_token):
                                should_merge = True
                                merged_text = combined_decoded
                        # 또는 두 토큰이 모두 짧고 합친 결과가 더 긴 경우
                        elif (
                            len(current_token) <= 1
                            and len(next_token) <= 1
                            and len(combined_decoded) > 1
                        ):
                            should_merge = True
                            merged_text = combined_decoded
            except Exception:
                pass

        if should_merge:
            # 병합: 합친 토큰 사용, attribution은 평균값
            merged_tokens.append(merged_text)
            merged_attributions.append((current_attr + attributions[i + 1]) / 2.0)
            # 원본 텍스트에서 병합된 텍스트 길이만큼 이동
            if text_pos < len(full_text):
                # 원본 텍스트에서 병합된 텍스트를 찾아서 위치 업데이트
                if full_text[text_pos:].startswith(merged_text):
                    text_pos += len(merged_text)
                else:
                    # 정확히 매칭되지 않으면 현재 토큰 길이만큼만 이동
                    text_pos += len(current_token) + len(decoded_tokens[i + 1])
            i += 2  # 두 토큰을 건너뜀
        else:
            # 병합하지 않는 경우 현재 토큰만 추가
            merged_tokens.append(current_token)
            merged_attributions.append(current_attr)
            # 원본 텍스트에서 현재 토큰 길이만큼 이동
            if text_pos < len(full_text):
                if full_text[text_pos:].startswith(current_token):
                    text_pos += len(current_token)
                else:
                    # 정확히 매칭되지 않으면 토큰 길이만큼만 이동
                    text_pos += len(current_token)
            i += 1

    return merged_tokens, merged_attributions


def visualize_attribution(result, tokenizer, max_attr=None, num_tokens_per_line=None):
    """Visualize token attributions with color boxes from white to dark orange."""
    # Extract tokens and attribution values
    tokens = [token for token, attr in result]
    attributions = [float(attr.item() if hasattr(attr, "item") else attr) for token, attr in result]

    # notebook / server 모두에서 동일한 토큰 병합 로직을 쓰도록 분리한 함수 사용
    decoded_tokens, attributions = decode_and_merge_tokens(tokens, attributions, tokenizer)
    
    # Normalize attributions to 0-1 range
    min_attr = min(attributions)
    if max_attr is None:
        max_attr = max(attributions)
    if max_attr > min_attr:
        normalized = [(a - min_attr) / (max_attr - min_attr) for a in attributions]
    else:
        normalized = [0.0] * len(attributions)
    
    # Create color map: white to dark orange
    # RGB: white (255, 255, 255) to dark orange (255, 140, 0)
    def get_color_html(value):
        # value is 0-1, map to white -> orange gradient
        r = 255
        g = int(255 - (255 - 140) * value)  # 255 -> 140
        b = int(255 - 255 * value)  # 255 -> 0
        return f"rgb({r}, {g}, {b})"

    # Generate HTML with better font support for Korean
    # UTF-8 인코딩을 명시적으로 지정
    html_parts = ['<div style="font-family: \'Nanum Gothic\', \'Malgun Gothic\', \'AppleGothic\', \'Noto Sans CJK KR\', sans-serif; line-height: 2.5; font-size: 14px;">']
    
    # num_tokens_per_line이 지정된 경우 줄바꿈 처리
    if num_tokens_per_line is not None and num_tokens_per_line > 0:
        for i, (token_text, norm_val) in enumerate(zip(decoded_tokens, normalized)):
            # 줄바꿈이 필요한 경우
            if i > 0 and i % num_tokens_per_line == 0:
                html_parts.append('<br>')
            
            color = get_color_html(norm_val)
            # Escape HTML special characters (한국어는 그대로 유지)
            # html.escape()는 ASCII가 아닌 문자를 그대로 유지합니다
            token_escaped = html.escape(token_text, quote=False)
            html_parts.append(
                f'<span style="background-color: {color}; padding: 5px 8px; margin: 2px; '
                f'border: 1px solid #ccc; border-radius: 3px; display: inline-block; color: black; '
                f'white-space: nowrap;">{token_escaped}</span>'
            )
    else:
        # num_tokens_per_line이 지정되지 않은 경우 기존처럼 한 줄에 모두 출력
        for token_text, norm_val in zip(decoded_tokens, normalized):
            color = get_color_html(norm_val)
            # Escape HTML special characters (한국어는 그대로 유지)
            token_escaped = html.escape(token_text, quote=False)
            html_parts.append(
                f'<span style="background-color: {color}; padding: 5px 8px; margin: 2px; '
                f'border: 1px solid #ccc; border-radius: 3px; display: inline-block; color: black; '
                f'white-space: nowrap;">{token_escaped}</span>'
            )
    
    html_parts.append('</div>')
    
    # UTF-8로 인코딩된 HTML 문자열 생성
    html_str = ''.join(html_parts)
    # HTML 객체에 UTF-8 인코딩 명시
    display(HTML(html_str))
    
    
import random
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def train_prober_with_L1_loss(prober, train_hiddens, train_labels, test_hiddens, test_labels, 
                  epochs, eval_every, batch_size, lr, L1_lambda=0.01, verbose=False, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prober.to(device)
    optimizer = torch.optim.Adam(prober.parameters(), lr=lr)
    train_dataset = TensorDataset(train_hiddens, train_labels)
    test_dataset = TensorDataset(test_hiddens, test_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    losses = []
    train_accs = []
    test_accs = []
    prober.train()
    eval_steps = []
    if verbose:
        pbar = tqdm(range(epochs))
    else:
        pbar = range(epochs)
        
    best_performance = 0
    best_prober = {k: v.detach().cpu().clone() for k, v in prober.state_dict().items()}

    thresholds = [] 
    l1_losses = []

    for epoch in pbar:
        all_outputs = [] 
        train_labels = []
        avg_loss = 0
        avg_l1_loss = 0
        for batch in train_dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = prober.forward(X)
            loss = prober.compute_loss(outputs, y)
            # L1 regularization on weights (exclude biases)
            # L1 regularization on weights (exclude biases)
            l1_params = [
                param for name, param in prober.named_parameters()
                if param.requires_grad and 'bias' not in name
            ]
            if l1_params:  # avoid division by zero
                l1_reg = torch.mean(torch.stack([p.abs().mean() for p in l1_params]))
                l1_loss = L1_lambda * l1_reg
                loss = loss + l1_loss
                avg_l1_loss += l1_loss.item()
                
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
            all_outputs.append(outputs)
            train_labels.append(y)
            
        all_outputs = torch.cat(all_outputs).reshape(-1)
        train_labels = torch.cat(train_labels).reshape(-1)
        # best threshold
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            threshold, train_acc = best_threshold_accuracy(all_outputs, train_labels)
            prober.threshold = threshold
            thresholds.append(threshold)
            test_acc = evaluate_prober(prober, train_hiddens, train_labels, 
                                                                   test_hiddens, test_labels, device)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            eval_steps.append(epoch + 1)
            avg_loss /= len(train_dataloader)
            avg_l1_loss /= len(train_dataloader)
            losses.append(avg_loss)
            l1_losses.append(avg_l1_loss)
            
            if test_acc > best_performance:
                best_performance = test_acc
                best_prober = {k: v.detach().cpu().clone() for k, v in prober.state_dict().items()}
            # pbar.set_postfix(train_acc=train_acc, test_acc=test_acc, threshold=threshold)
            if verbose:
                print(f"Epoch {epoch + 1}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Threshold: {threshold:.4f}")
                
    prober.load_state_dict(best_prober)
    return prober, losses, l1_losses, train_accs, test_accs, eval_steps, thresholds


def train_prober(prober, train_hiddens, train_labels, test_hiddens, test_labels, 
                  epochs, eval_every, batch_size, lr, verbose=False, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prober.to(device)
    optimizer = torch.optim.Adam(prober.parameters(), lr=lr)
    train_dataset = TensorDataset(train_hiddens, train_labels)
    test_dataset = TensorDataset(test_hiddens, test_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    losses = []
    train_accs = []
    test_accs = []
    prober.train()
    eval_steps = []
    if verbose:
        pbar = tqdm(range(epochs))
    else:
        pbar = range(epochs)
        
    best_performance = 0
    best_prober = {k: v.detach().cpu().clone() for k, v in prober.state_dict().items()}
    
    thresholds = [] 

    for epoch in pbar:
        all_outputs = [] 
        train_labels = []
        avg_loss = 0
        for batch in train_dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = prober.forward(X)
            loss = prober.compute_loss(outputs, y)
            
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
            all_outputs.append(outputs)
            train_labels.append(y)
            
        all_outputs = torch.cat(all_outputs).reshape(-1)
        train_labels = torch.cat(train_labels).reshape(-1)
        # best threshold
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            threshold, train_acc = best_threshold_accuracy(all_outputs, train_labels)
            prober.threshold = threshold
            thresholds.append(threshold)
            test_acc = evaluate_prober(prober, train_hiddens, train_labels, 
                                                                   test_hiddens, test_labels, device)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            eval_steps.append(epoch + 1)
            avg_loss /= len(train_dataloader)
            losses.append(avg_loss)
            
            if test_acc > best_performance:
                best_performance = test_acc
                best_prober = {k: v.detach().cpu().clone() for k, v in prober.state_dict().items()}
            # pbar.set_postfix(train_acc=train_acc, test_acc=test_acc, threshold=threshold)
            if verbose:
                print(f"Epoch {epoch + 1}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Threshold: {threshold:.4f}")
                
    prober.load_state_dict(best_prober)
    return prober, losses, train_accs, test_accs, eval_steps, thresholds




def evaluate_prober(prober, train_hiddens, train_labels, test_hiddens, test_labels, device):
    train_dataset = TensorDataset(train_hiddens, train_labels)
    test_dataset = TensorDataset(test_hiddens, test_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    is_train = True if train_dataloader else False
    prober.eval()
    with torch.no_grad():
        # find the best threshold with train set
        # train_accs = []
        test_accs = []
        # for batch in train_dataloader:
        #     X, y = batch
        #     X = X.to(device)
        #     y = y.to(device)
        #     train_preds, _ = prober.predict(X, threshold=prober.threshold)
        #     train_acc = (train_preds == y).float()
        #     train_accs.append(train_acc)
            
        for batch in test_dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            test_preds, _ = prober.predict(X, threshold=prober.threshold)
            test_acc = (test_preds == y).float()
            test_accs.append(test_acc)
    # train_accs = torch.cat(train_accs)
    # train_acc = train_accs.mean() 
    test_accs = torch.cat(test_accs)
    test_acc = test_accs.mean()
    
    # train_acc = train_acc.item()
    test_acc = test_acc.item()
    prober.train() if is_train else prober.eval()
    
    # return train_acc, test_acc
    return test_acc
    

def best_threshold_accuracy(y_score: torch.Tensor, y_true: torch.Tensor):
    assert y_score.ndim == 1 and y_true.ndim == 1 and y_score.size(0) == y_true.size(0)

    scores, idx = torch.sort(y_score, descending=True)
    labels = y_true[idx].to(torch.long)

    tp_cum = torch.cumsum(labels, dim=0)
    fp_cum = torch.cumsum(1 - labels, dim=0)
    P = labels.sum()
    N = labels.numel() - P

    tn = N - fp_cum
    tp = tp_cum

    acc = (tp + tn).float() / (P + N)

    best_idx = torch.argmax(acc)
    if best_idx < scores.numel() - 1:
        thr = 0.5 * (scores[best_idx] + scores[best_idx + 1])
    else:
        thr = scores[best_idx]

    return float(thr), float(acc[best_idx])


# ==============================

MODELS = [    
    'meta-llama/Llama-3.3-70B-Instruct',  # https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
    'meta-llama/Llama-3.1-8B-Instruct',  # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    
    'meta-llama/Llama-4-Scout-17B-16E-Instruct',  # https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
    'meta-llama/Llama-4-Maverick-17B-128E-Instruct',  # https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct
    
    'google/gemma-3-4b-it',   # https://huggingface.co/google/gemma-3-4b-it
    'google/gemma-3-12b-it', # https://huggingface.co/google/gemma-3-12b-pt
    'Qwen/Qwen2.5-7B-Instruct', # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
    'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct', # https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
]

import torch 
import datasets 
from transformers import AutoTokenizer, AutoModelForCausalLM

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


def get_num_layers(llm_name):
    if llm_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        return 32
    elif llm_name == "Qwen/Qwen2.5-7B-Instruct":
        return 28
    else:
        raise ValueError(f"Unsupported model: {llm_name}")

def get_mlp_down_proj(llm_name, block):
    if 'meta-llama' in llm_name:
        module = block.mlp.down_proj
    elif 'Qwen' in llm_name:
        module = block.mlp.down_proj
    else:
        raise ValueError(f"Unsupported model: {llm_name}")
    return module

def get_mlp_up_proj(llm_name, block):
    if 'meta-llama' in llm_name:
        module = block.mlp.up_proj
    elif 'Qwen' in llm_name:
        module = block.mlp.up_proj
    else:
        raise ValueError(f"Unsupported model: {llm_name}")
    return module


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in ("true", "t", "1", "yes", "y", "on"):
        return True
    if val in ("false", "f", "0", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean value for only_last_token: {value}")



import json 
def load_attribution_weight(attribution_method, model_name, dataset_name, prompt_type, only_last_token, init_seed=None):
    attribution_names = ['by_act', 'by_random', 'by_lrp', 'from_value_delta', 'from_value_only']
    assert attribution_method in attribution_names, f"Invalid attribution method: {attribution_method}"
    seed_attribution_names = ['by_lrp', 'from_value_delta', 'from_value_only']
    if attribution_method in seed_attribution_names:
        result_format = "outputs/attribution/{attribution_name}/{model_name}/{dataset_name}/{prompt_type}/only_last_token_{only_last_token}/{seed}/attribution.json"
        attribution_weight_dict = json.load(open(result_format.format(attribution_name=attribution_method, 
                                                                model_name=model_name, 
                                                                dataset_name=dataset_name, 
                                                                prompt_type=prompt_type, 
                                                                only_last_token=only_last_token, 
                                                                seed=init_seed)))
    else:
        result_format = "outputs/attribution/{attribution_name}/{model_name}/{dataset_name}/{prompt_type}/only_last_token_{only_last_token}/attribution.json"
        attribution_weight_dict = json.load(open(result_format.format(attribution_name=attribution_method, 
                                                                      model_name=model_name, 
                                                                      dataset_name=dataset_name, 
                                                                      prompt_type=prompt_type, 
                                                                      only_last_token=only_last_token, 
                                                                      )))
    
    return attribution_weight_dict