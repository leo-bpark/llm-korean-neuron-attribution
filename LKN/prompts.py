from typing import Iterable, Dict, Any, Union
import torch
from torch.utils.data import DataLoader

# ========================================================================================
def formatting_prompts(
    model_name: str,
    dataset: Iterable[Dict[str, Any]],
    tokenizer,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
):
    
    
    """
    Preprocess a dataset of samples into tokenized, probe-ready batches.

    Input dataset: iterable of dicts with at least {"text": <str>}
    Side-effects per sample:
      - Replaces "text" with the finalized user text actually used
      - Adds "input_ids" (list[int])
      - Adds "probe_positions" (list[int])

    Returns:
      torch.utils.data.DataLoader that yields dict with:
        - input_ids:     (B, T) padded LongTensor
        - attention_mask:(B, T) LongTensor
        - probe_mask:    (B, T) BoolTensor, True where to extract hidden states
        - seq_lens:      (B,)   LongTensor, original sequence lengths
    """
    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        # set pad token id to eos if missing (common for some LLMs)
        if getattr(tokenizer, "eos_token_id", None) is None:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token_id; set one before batching.")
        tokenizer.pad_token = tokenizer.eos_token

    processed: List[Dict[str, Any]] = []
    
    for sample in dataset:
        sample_text = sample["text"]
        sample_id = sample["sample_id"]

        out = prepare_probe_inputs(
            model_name=model_name,
            sample_text=sample_text,
            tokenizer=tokenizer,
            use_chat_template=True,
        )

        processed.append({
            # keep anything else the original sample had
            **sample,
            "text": out["final_text"],
            "input_ids": out["input_ids"],
            "probe_positions": out["probe_positions"],
            "sample_ids": sample_id,
            "labels": sample["label"],
        })

    # Simple list-backed dataset (so DataLoader can index it)
    class _ListDataset(torch.utils.data.Dataset):
        def __init__(self, data: List[Dict[str, Any]]):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    ds = _ListDataset(processed)
    collate_fn = lambda batch: _collate_with_probe_mask(batch, tokenizer.pad_token_id)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

# ========================================================================================
from typing import Dict, List, Any, Iterable
from torch.utils.data import DataLoader
import torch

def _collate_with_probe_mask(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """
    Collate function that:
      - Pads input_ids to the max length in the batch (LEFT PADDING)
      - Builds attention_mask
      - Builds probe_mask (bool), True at probe positions per sample
      - Also returns (optional) per-sample lengths for convenience
    """
    # lengths
    lengths = [len(ex["input_ids"]) for ex in batch]
    max_len = max(lengths)

    # allocate tensors
    input_ids = torch.full((len(batch), max_len), fill_value=pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    probe_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    sample_ids = torch.tensor([ex['sample_ids'] for ex in batch], dtype=torch.long)
    labels = torch.tensor([ex['label'] for ex in batch], dtype=torch.long)
    # Collect all probe positions for each sample (will be padded to max_len)
    all_probe_positions = []
    
    for i, ex in enumerate(batch):
        ids = torch.tensor(ex["input_ids"], dtype=torch.long)
        L = ids.shape[0]
        
        # LEFT PADDING: place sequence at the end (right side) of the tensor
        pad_start = max_len - L
        input_ids[i, pad_start:] = ids
        attention_mask[i, pad_start:] = 1

        # Adjust probe positions for left padding offset
        adjusted_probe_positions = []
        for p in ex.get("probe_positions", []):
            if 0 <= p < L:
                # Add the padding offset to get the new position
                adjusted_pos = pad_start + p
                probe_mask[i, adjusted_pos] = True
                adjusted_probe_positions.append(adjusted_pos)
        
        # Pad probe positions to max_len with -1 (invalid position)
        padded_probe_pos = []
        for j, pos in enumerate(adjusted_probe_positions):
            if j < max_len:
                padded_probe_pos.append(pos)
        all_probe_positions.append(padded_probe_pos)

    probe_positions = torch.tensor(all_probe_positions, dtype=torch.long)

    return {
        "input_ids": input_ids,            # (B, T)
        "attention_mask": attention_mask,  # (B, T)
        "probe_mask": probe_mask,          # (B, T) bool
        "probe_positions": probe_positions,  # (B, T) int
        "sample_ids": sample_ids,          # (B,)
        "seq_lens": torch.tensor(lengths, dtype=torch.long),  # (B,)
        "labels": labels,          # (B,)
    }

def _decode_tokens(tokenizer, input_ids: List[int]) -> List[str]:
    # Best-effort piecewise decoding (works for fast tokenizers)
    # If you prefer, you can just return tokenizer.convert_ids_to_tokens(input_ids)
    try:
        return tokenizer.convert_ids_to_tokens(input_ids)
    except Exception:
        return [tokenizer.decode([tid], skip_special_tokens=False) for tid in input_ids]


# ========================================================================================
def prepare_probe_inputs(
    model_name: str,
    sample_text: str,
    tokenizer,
    use_chat_template: bool = True,
) -> Dict[str, Union[str, List[int], List[str], List[int]]]:
    """
    Build the final (chat-formatted) text, tokenize it, and compute probe positions.

    Returns:
        {
          "final_text": str,              # the final user text used in messages (pre-template)
          "chat_formatted_text": str,     # printable chat-formatted text (if available; else same as final_text)
          "input_ids": List[int],         # tokenized ids of the fully formatted prompt
          "tokens": List[str],            # token pieces
          "probe_positions": List[int],   # indices at which to extract hidden states
        }
    """

    # 1) Build the *user* text according to condition
    user_text = sample_text

    # 2) Build chat-formatted input (messages → template → input_ids)
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        if 'meta-llama' in model_name:
            messages = [{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": user_text}]
        elif 'Qwen' in model_name:
            messages = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}, {"role": "user", "content": user_text}]
        
        # tokenize=True returns input_ids; return_tensors=None keeps them as a list
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True
        )
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
        chat_formatted_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
    else:
        # Fallback: simple raw encode
        chat_formatted_text = user_text
        input_ids = tokenizer.encode(
            user_text,
            add_special_tokens=True
        )

    # 3) Compute probe positions
    probe_positions: List[int] = []

    n = len(input_ids)
    probe_positions = [n - 5, n - 4, n - 3, n - 2, n - 1]

    # 4) Pack results
    tokens = _decode_tokens(tokenizer, input_ids)

    return {
        "final_text": user_text,                 # the user content used to build messages
        "chat_formatted_text": chat_formatted_text,  # printable form of the full prompt
        "input_ids": input_ids,
        "probe_positions": probe_positions,
        "tokens": tokens,
    }