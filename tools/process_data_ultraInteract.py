import multiprocessing
import os
import time
import torch
import json
import sys
import random
import numpy as np
from data_utils.indexed_dataset import make_builder
from transformers import AutoTokenizer
from arguments import get_args

random.seed(42)

# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

    def encode(self, line):
        raw_prompt = line["prompt"]
        response = line["generated_text"]

        messages = [
            {"role": "user", "content": raw_prompt}
        ]
        
        prompt_str = Encoder.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        prompt_tokens = Encoder.tokenizer.encode(prompt_str, add_special_tokens=False)
        full_tokens = Encoder.tokenizer.encode(prompt_str + response, add_special_tokens=False) + [
            Encoder.tokenizer.eos_token_id]

        response_tokens = full_tokens[len(prompt_tokens):]

        if len(prompt_tokens) > self.args.max_prompt_length:
            prompt_tokens = prompt_tokens[:self.args.max_prompt_length]
            
        bytes_processed = len(prompt_str.encode('utf-8')) + len(response.encode('utf-8'))

        return line, prompt_str, prompt_tokens, response_tokens, bytes_processed


def main():
    print("OK")
    args = get_args()

    args.processed_data_dir = os.path.join(args.processed_data_dir, args.model_path)

    os.makedirs(args.processed_data_dir, exist_ok=True)

    raw_data = []
    print(f"Reading data from: {args.data_dir}")
    with open(args.data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
    print(f"Total data instances: {len(raw_data)}")
    
    if args.dev_num > 0:
        valid_data = random.sample(raw_data, args.dev_num)
        all_data = {
            "valid": valid_data,
            "train": raw_data
        }
    else:
        all_data = {
            "train": raw_data
        }

    for split in all_data:

        # encoder use the tokenizer to encode data
        encoder = Encoder(args)

        # 2. Mapping all datas with Encoder, with the help of multiprocessing
        pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
        # Preserve the canonical raw JSONL order. Exact dataset reproduction
        # requires more than an equal sample set: every trainer must see the
        # same sequence before its seeded sampler is applied.
        encoded_docs = pool.imap(encoder.encode, all_data[split], chunksize=50)
        proc_start = time.time()
        total_bytes_processed = 0

        bin_file = os.path.join(args.processed_data_dir, f"{split}_{0}.bin")
        idx_file = os.path.join(args.processed_data_dir, f"{split}_{0}.idx")

        binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint32)

        # put tokenized data into binary_builder
        inst_num = 0
        print("#" * 10, split, "#" * 10)

        prompt_lens = []
        response_lens = []

        json_file = open(os.path.join(args.processed_data_dir, f"{split}.jsonl"), "w")

        for lid, (line, prompt_str, prompt, response, bytes_processed) in enumerate(encoded_docs):
            total_bytes_processed += bytes_processed
            if prompt is None:
                continue

            if args.only_prompt:
                if len(prompt) < args.max_length:
                    binary_builder.add_item(torch.IntTensor(prompt))
                else:
                    continue
            else:
                binary_builder.add_item(torch.IntTensor(prompt + [-1] + response))

            json_file.write(json.dumps({
                "instruction": line["prompt"],
                "prompt": prompt_str,
                "output": line["generated_text"]
            }) + "\n")

            prompt_lens.append(len(prompt))
            response_lens.append(len(response))

            inst_num += 1
            if lid % 1000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {lid} documents. {inst_num} instances.",
                      f"({lid / elapsed:.2f} docs/s, {mbs:.4f} MB/s).",
                      file=sys.stderr)

        # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
        binary_builder.finalize(idx_file)

        # close multiproceessing mapping
        pool.close()
        json_file.close()

        print("Data num", len(prompt_lens))
        print("Prompt lengths.", "Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:",
              np.min(prompt_lens))
        print("Response", "Mean:", np.mean(response_lens), "Max:", np.max(response_lens), "Min:", np.min(response_lens))


if __name__ == '__main__':
    main()
