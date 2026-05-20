"""
Evaluation script for eval-input.tsv

Format: each line has 2 tab-separated sentences. The correct sentence is unknown.
The model assigns higher probability to what it considers the correct sentence,
and writes predictions to eval_output.txt.

Usage (standard checkpoint):
    python -m scripts.eval_test --model-tag d6 --test-path ../eval-input.tsv

Usage (custom checkpoint):
    python -m scripts.eval_test --checkpoint-path ./data/base_checkpoints/tuned --test-path ../eval-input.tsv

Usage (with GPT-2 ground truth comparison):
    python -m scripts.eval_test --model-tag d6 --test-path ../eval-input.tsv --gpt2-eval

With --gpt2-eval, GPT-2 predictions are printed to stdout and treated as ground truth
to measure accuracy of the main model against them.
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


# ---------------------------------------------------------------------------
# Checkpoint helpers (shared with eval_devel)
# ---------------------------------------------------------------------------

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

    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)

    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

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


# ---------------------------------------------------------------------------
# Log-prob helpers
# ---------------------------------------------------------------------------

def compute_sentence_log_prob(model, tokenizer, sentence, device):
    """
    Compute the average log probability per token for a sentence using
    the nanochat model.

    Returns:
        avg_log_prob (float)
    """
    tokens = tokenizer(sentence, prepend="<|bos|>")
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    with torch.no_grad():
        logits = model(tokens, targets=None)  # (1, T, vocab_size)

    log_probs = torch.log_softmax(logits, dim=-1)  # (1, T, vocab_size)
    target_tokens = tokens[:, 1:]  # (1, T-1)
    selected_log_probs = torch.gather(
        log_probs[:, :-1, :],
        dim=2,
        index=target_tokens.unsqueeze(-1),
    ).squeeze(-1)  # (1, T-1)

    return selected_log_probs.mean().item()


def compute_sentence_log_prob_gpt2(gpt2_model, gpt2_tokenizer, sentence, device):
    """
    Compute the average log probability per token for a sentence using
    a HuggingFace GPT-2 model.

    Returns:
        avg_log_prob (float)
    """
    enc = gpt2_tokenizer(sentence, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # (1, T)

    with torch.no_grad():
        outputs = gpt2_model(input_ids, labels=input_ids)
        # outputs.loss is the mean negative log-likelihood per token
        avg_nll = outputs.loss.item()

    return -avg_nll  # convert NLL → log-prob (higher = more likely)


# ---------------------------------------------------------------------------
# GPT-2 loader
# ---------------------------------------------------------------------------

def load_gpt2_model(device):
    """Load GPT-2 from HuggingFace transformers."""
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    except ImportError:
        raise ImportError(
            "The 'transformers' package is required for --gpt2-eval. "
            "Install it with: pip install transformers"
        )

    print0("Loading GPT-2 from HuggingFace transformers...")
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.to(device)
    gpt2_model.eval()
    print0("✓ GPT-2 loaded")
    return gpt2_model, gpt2_tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Predict correct sentences on eval-input.tsv")
    parser.add_argument("--model-tag", type=str, default="d6",
                        help="checkpoint tag (e.g. 'd6', 'd24')")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="custom checkpoint directory path (overrides --model-tag)")
    parser.add_argument("--test-path", type=str, default="data/datasets/eval-input.tsv",
                        help="path to test .tsv file")
    parser.add_argument("--output-path", type=str, default="data/datasets/eval_output.txt",
                        help="path to write predictions (default: eval_output.txt)")
    parser.add_argument("--device-type", type=str, default="",
                        help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--max-examples", type=int, default=-1,
                        help="max examples to evaluate (-1 = all)")
    parser.add_argument("--gpt2-eval", action="store_true",
                        help=(
                            "Also evaluate with GPT-2 as a reference model. "
                            "GPT-2 predictions are printed to stdout and used as "
                            "ground truth to score the main model."
                        ))
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Device setup
    # ------------------------------------------------------------------
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    print0(f"Device: {device} ({device_type})")

    # ------------------------------------------------------------------
    # Load main model
    # ------------------------------------------------------------------
    if args.checkpoint_path:
        print0(f"Loading model from custom checkpoint: {args.checkpoint_path}...")
        model, tokenizer, meta_data = load_custom_checkpoint(args.checkpoint_path, device)
    else:
        print0(f"Loading model with tag '{args.model_tag}'...")
        model, tokenizer, meta_data = load_model("base", device, phase="eval", model_tag=args.model_tag)
    print0("✓ Main model loaded")
    print0("✓ Tokenizer loaded")

    # ------------------------------------------------------------------
    # Optionally load GPT-2
    # ------------------------------------------------------------------
    gpt2_model = gpt2_tokenizer_hf = None
    if args.gpt2_eval:
        gpt2_model, gpt2_tokenizer_hf = load_gpt2_model(device)

    # ------------------------------------------------------------------
    # Read test file
    # ------------------------------------------------------------------
    if not os.path.exists(args.test_path):
        print0(f"Error: {args.test_path} not found")
        return

    with open(args.test_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print0(f"Loaded {len(lines)} lines from {args.test_path}")

    max_examples = len(lines) if args.max_examples == -1 else args.max_examples

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    predictions = []      # main model: 0 = first sentence preferred, 1 = second
    gpt2_predictions = [] # gpt2: same convention

    for i, line in enumerate(tqdm(lines[:max_examples], desc="Evaluating")):
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) != 2:
            print0(f"Warning: skipping malformed line {i+1}: {line}")
            continue

        sent_a, sent_b = parts

        # ---- Main model ------------------------------------------------
        lp_a = compute_sentence_log_prob(model, tokenizer, sent_a, device)
        lp_b = compute_sentence_log_prob(model, tokenizer, sent_b, device)
        main_pred = 0 if lp_a >= lp_b else 1
        predictions.append((sent_a, sent_b, main_pred, lp_a, lp_b))

        # ---- GPT-2 reference -------------------------------------------
        if gpt2_model is not None:
            gpt2_lp_a = compute_sentence_log_prob_gpt2(gpt2_model, gpt2_tokenizer_hf, sent_a, device)
            gpt2_lp_b = compute_sentence_log_prob_gpt2(gpt2_model, gpt2_tokenizer_hf, sent_b, device)
            gpt2_pred = 0 if gpt2_lp_a >= gpt2_lp_b else 1
            gpt2_predictions.append(gpt2_pred)

            chosen_gpt2 = sent_a if gpt2_pred == 0 else sent_b
            # print(
            #     f"[GPT-2] Example {i+1}: "
            #     f"lp(A)={gpt2_lp_a:.4f}  lp(B)={gpt2_lp_b:.4f}  "
            #     f"→ prefers {'A' if gpt2_pred == 0 else 'B'}: {chosen_gpt2[:80]}"
            # )
            
            disagreement = main_pred != gpt2_pred
            
            if disagreement:
                print(
                    f"[DISAGREE] Example {i+1}: "
                    f"Main model prefers {'A' if main_pred == 0 else 'B'} (lp(A)={lp_a:.4f}, lp(B)={lp_b:.4f}), "
                    f"GPT-2 prefers {'A' if gpt2_pred == 0 else 'B'} (lp(A)={gpt2_lp_a:.4f}, lp(B)={gpt2_lp_b:.4f})\n"
                    f"    A: {sent_a}\n"
                    f"    B: {sent_b}\n"
                )

    # ------------------------------------------------------------------
    # Write main-model predictions to file
    # ------------------------------------------------------------------
    with open(args.output_path, "w", encoding="utf-8") as out:
        for sent_a, sent_b, pred, lp_a, lp_b in predictions:
            chosen = sent_a if pred == 0 else sent_b
            out.write(chosen + "\n")

    print0(f"\n✓ Predictions written to {args.output_path}  ({len(predictions)} lines)")

    # ------------------------------------------------------------------
    # If GPT-2 was used, score main model against GPT-2 as ground truth
    # ------------------------------------------------------------------
    if gpt2_predictions:
        assert len(gpt2_predictions) == len(predictions), "Length mismatch between model and GPT-2 predictions"
        agree = sum(
            1 for (_, _, main_pred, _, _), gpt2_pred in zip(predictions, gpt2_predictions)
            if main_pred == gpt2_pred
        )
        total = len(gpt2_predictions)
        acc = 100.0 * agree / total if total > 0 else 0.0
        print0(f"\n{'='*60}")
        print0(f"Agreement with GPT-2 ground truth on {total} examples:")
        print0(f"Accuracy: {agree}/{total} = {acc:.2f}%")
        print0(f"{'='*60}\n")

    compute_cleanup()


if __name__ == "__main__":
    main()