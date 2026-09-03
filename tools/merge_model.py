#!/usr/bin/env python3

import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import os
import safetensors.torch as st




def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA/DoRA adapter into base model"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model path or HF repo"
    )

    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to LoRA/DoRA adapter"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for merged model"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading base model: {args.base_model}")

    adapter_bin = os.path.join(args.adapter, "adapter_model.bin")
    adapter_safe = os.path.join(args.adapter, "adapter_model.safetensors")

    if os.path.exists(adapter_safe):
        adapter_weights = st.load_file(adapter_safe)
    else:
        adapter_weights = torch.load(adapter_bin, map_location="cpu")

    vocab_size = -1
    if 'model.embed_tokens.weight' in adapter_weights:
        vocab_size = adapter_weights['model.embed_tokens.weight'].shape[0]
    elif 'base_model.model.model.embed_tokens.weight' in adapter_weights:
        vocab_size = adapter_weights["base_model.model.model.embed_tokens.weight"].shape[0]

    print(f"Detected vocab size from adapter: {vocab_size}")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    if vocab_size > 0:
        base_model.resize_token_embeddings(vocab_size)

    print(f"Loading adapter: {args.adapter}")

    model = PeftModel.from_pretrained(
        base_model,
        args.adapter,
    )

    print("Merging adapter into base model...")

    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output}")

    merged_model.save_pretrained(args.output)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    tokenizer.save_pretrained(args.output)

    print("Done!")


if __name__ == "__main__":
    main()