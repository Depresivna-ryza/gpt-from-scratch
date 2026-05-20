"""
Simple evaluation script for devel.tsv.txt

Format: each line has 2 tab-separated sentences. First is correct, second is incorrect.
The model should assign higher probability to the first sentence.

Usage (standard checkpoint):
    python -m scripts.eval_devel --model-tag d6 --devel-path ../devel.tsv.txt

Usage (custom checkpoint, e.g., from grammar_preference_tune):
    python -m scripts.eval_devel --checkpoint-path ./data/base_checkpoints/tuned --devel-path ../devel.tsv.txt

Measures accuracy: what fraction of the time does the model prefer the correct sentence?
"""

import os
import argparse
import glob
import json
import torch
from tqdm import tqdm

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest model checkpoint in a directory by step number."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

def load_custom_checkpoint(checkpoint_dir, device):
    """Load a model from a custom checkpoint directory (e.g., from grammar_preference_tune)."""
    step = find_latest_checkpoint(checkpoint_dir)
    
    # Load model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    
    # Load metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    
    # Build model from config
    model_config_kwargs = meta_data.get("model_config", {})
    if not model_config_kwargs:
        raise ValueError(f"No model_config found in {meta_path}")
    
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    model_config_kwargs["vocab_size"] = vocab_size
    
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()
    
    return model, tokenizer, meta_data

def compute_sentence_log_prob(model, tokenizer, sentence, device):
    """
    Compute the average log probability per token for a sentence.
    
    Args:
        model: the language model
        tokenizer: tokenizer
        sentence: string
        device: device to run on
    
    Returns:
        avg_log_prob: average log probability per token
    """
    # Tokenize with BOS token
    tokens = tokenizer(sentence, prepend="<|bos|>")
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    
    # Forward pass
    with torch.no_grad():
        logits = model(tokens, targets=None)  # (1, T, vocab_size)
    
    # Compute log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)  # (1, T, vocab_size)
    
    # Get the log prob of the actual next token
    target_tokens = tokens[:, 1:]  # (1, T-1)
    selected_log_probs = torch.gather(
        log_probs[:, :-1, :],  # (1, T-1, vocab_size)
        dim=2,
        index=target_tokens.unsqueeze(-1)  # (1, T-1, 1)
    ).squeeze(-1)  # (1, T-1)
    
    # Average log prob per token
    avg_log_prob = selected_log_probs.mean().item()
    return avg_log_prob


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on devel.tsv.txt")
    parser.add_argument("--model-tag", type=str, default="d4", help="checkpoint tag (e.g. 'd6', 'd24')")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="custom checkpoint directory path (overrides --model-tag)")
    # parser.add_argument("--devel-path", type=str, default="data/datasets/devel.tsv", help="path to devel.tsv.txt")
    parser.add_argument("--devel-path", type=str, default="data/datasets/eval_val_split.tsv", help="path to devel.tsv.txt")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--max-examples", type=int, default=-1, help="max examples to evaluate (-1 = all)")
    args = parser.parse_args()
    
    # Setup device
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    
    print0(f"Device: {device} ({device_type})")
    
    # Load model and tokenizer
    if args.checkpoint_path:
        print0(f"Loading model from custom checkpoint: {args.checkpoint_path}...")
        model, tokenizer, meta_data = load_custom_checkpoint(args.checkpoint_path, device)
    else:
        print0(f"Loading model with tag '{args.model_tag}'...")
        model, tokenizer, meta_data = load_model("base", device, phase="eval", model_tag=args.model_tag)
    print0(f"✓ Model loaded")
    print0(f"✓ Tokenizer loaded")
    
    # Read devel.tsv.txt
    if not os.path.exists(args.devel_path):
        print0(f"Error: {args.devel_path} not found")
        return
    
    with open(args.devel_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print0(f"Loaded {len(lines)} examples from {args.devel_path}")
    
    # Evaluate
    correct = 0
    total = 0
    
    max_examples = len(lines) if args.max_examples == -1 else args.max_examples
    
    for line in tqdm(lines[:max_examples], desc="Evaluating"):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('\t')
        if len(parts) != 2:
            print0(f"Warning: skipping malformed line: {line}")
            continue
        
        sent_correct, sent_incorrect = parts
        
        # Compute log probs
        with torch.no_grad():
            lp_correct = compute_sentence_log_prob(model, tokenizer, sent_correct, device)
            lp_incorrect = compute_sentence_log_prob(model, tokenizer, sent_incorrect, device)
        
        # Check if model prefers correct sentence
        if lp_correct > lp_incorrect:
            correct += 1
            
        # print(f"----------{total+1}:")
        # print(f"{sent_correct[:50]}")
        # print(f"{sent_incorrect[:50]}")
        # print(f"Correct: {lp_correct}, Incorrect: {lp_incorrect}")
        
        total += 1
    
    accuracy = 100 * correct / total if total > 0 else 0
    print0(f"\n{'='*60}")
    print0(f"Results on {total} examples from {args.devel_path}:")
    print0(f"Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print0(f"{'='*60}\n")
    
    # Cleanup
    compute_cleanup()


if __name__ == "__main__":
    main()
