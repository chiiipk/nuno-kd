"""Turn a teacher-generation JSONL into the trainer's tokenized format.

    python -m nuno_kd.data.prepare convert  --source RAW --target CANONICAL --expected N
    python -m nuno_kd.data.prepare tokenize --data-file CANONICAL --processed-data-dir ROOT \
        --model-path MODEL [--max-prompt-length 512] [--dev-num 200] [--workers 32]

`tokenize` writes to ROOT/MODEL:
- `{split}_0.bin/.idx`: `prompt + [SEPARATOR] + response + [eos]` token ids per example;
- `{split}.jsonl`: the matching instruction, chat-formatted prompt and response.
The `train` split keeps every example in source order; `valid` holds --dev-num
examples sampled from them with a fixed seed.
"""

import argparse
import json
import multiprocessing
import os
import random
import sys
import time

import numpy as np
from transformers import AutoTokenizer

from .indexed import IndexedTokensWriter

TOKEN_DTYPE = np.uint32
# Marks the prompt/response boundary inside a stored sequence; never a token id.
SEPARATOR = np.iinfo(TOKEN_DTYPE).max


def convert(source, target, expected):
    """instruction/response JSONL -> prompt/generated_text JSONL, in source order."""
    temporary = target + ".partial"
    count = 0
    with open(source, encoding="utf-8") as src, open(temporary, "w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt, response = row.get("instruction"), row.get("response")
            if not isinstance(prompt, str) or not isinstance(response, str):
                raise ValueError(f"{source}:{line_number}: expected string instruction/response")
            dst.write(json.dumps({"prompt": prompt, "generated_text": response}, ensure_ascii=False) + "\n")
            count += 1
    if count != expected:
        raise SystemExit(f"Expected {expected} examples, found {count}")
    os.replace(temporary, target)
    print(f"[converted] {count} ordered examples -> {target}")


class Encoder:
    """Tokenizes one example; one tokenizer per worker process."""

    def __init__(self, model_path, max_prompt_length):
        self.model_path = model_path
        self.max_prompt_length = max_prompt_length

    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def encode(self, row):
        tokenizer = Encoder.tokenizer
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        prompt = tokenizer.encode(prompt_str, add_special_tokens=False)
        full = tokenizer.encode(prompt_str + row["generated_text"], add_special_tokens=False)
        response = full[len(prompt):] + [tokenizer.eos_token_id]
        return row, prompt_str, prompt[:self.max_prompt_length], response


def tokenize(data_file, processed_data_dir, model_path, max_prompt_length, dev_num, workers):
    output_dir = os.path.join(processed_data_dir, model_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(data_file, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    print(f"Read {len(rows)} examples from {data_file}")
    splits = {"train": rows}
    if dev_num > 0:
        splits = {"valid": random.Random(42).sample(rows, dev_num), "train": rows}

    encoder = Encoder(model_path, max_prompt_length)
    for split, split_rows in splits.items():
        writer = IndexedTokensWriter(os.path.join(output_dir, f"{split}_0"), TOKEN_DTYPE)
        prompt_lens, response_lens = [], []
        started = time.time()
        with multiprocessing.Pool(workers, initializer=encoder.initializer) as pool, \
                open(os.path.join(output_dir, f"{split}.jsonl"), "w") as sidecar:
            # Ordered imap: the trainer must see the examples in source order.
            for i, (row, prompt_str, prompt, response) in enumerate(pool.imap(encoder.encode, split_rows, chunksize=50)):
                writer.add(prompt + [SEPARATOR] + response)
                sidecar.write(json.dumps({
                    "instruction": row["prompt"], "prompt": prompt_str, "output": row["generated_text"],
                }) + "\n")
                prompt_lens.append(len(prompt))
                response_lens.append(len(response))
                if i % 1000 == 0:
                    print(f"[{split}] {i} examples ({i / (time.time() - started):.1f}/s)", file=sys.stderr)
        writer.finalize()
        print(f"[{split}] {len(prompt_lens)} examples -> {output_dir}")
        print(f"  prompt length:   mean {np.mean(prompt_lens):.1f}, max {np.max(prompt_lens)}, min {np.min(prompt_lens)}")
        print(f"  response length: mean {np.mean(response_lens):.1f}, max {np.max(response_lens)}, min {np.min(response_lens)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    conv = sub.add_parser("convert")
    conv.add_argument("--source", required=True)
    conv.add_argument("--target", required=True)
    conv.add_argument("--expected", type=int, required=True)
    tok = sub.add_parser("tokenize")
    tok.add_argument("--data-file", required=True)
    tok.add_argument("--processed-data-dir", required=True)
    tok.add_argument("--model-path", required=True)
    tok.add_argument("--max-prompt-length", type=int, default=512)
    tok.add_argument("--dev-num", type=int, default=0)
    tok.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "convert":
        convert(args.source, args.target, args.expected)
    else:
        tokenize(args.data_file, args.processed_data_dir, args.model_path,
                 args.max_prompt_length, args.dev_num, args.workers)


if __name__ == "__main__":
    main()
