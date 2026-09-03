import os
import json
import gc
import math
import argparse
import multiprocessing as mp
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from vllm import LLM, SamplingParams

def worker_inference(gpu_id, model_path, data_chunk, prompts_chunk, temp_out_path):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import json
    import gc
    import math
    import argparse
    import multiprocessing as mp
    from transformers import AutoTokenizer
    from datasets import load_dataset
    import torch
    from vllm import LLM, SamplingParams
    
    print(f"\n[GPU {gpu_id}] 🚀 Khởi tạo model trên GPU {gpu_id}...")
    print(f"[GPU {gpu_id}] 📦 Số lượng prompt đảm nhiệm: {len(prompts_chunk)}")
    
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        seed=42
    )

    sampling_params = SamplingParams(
        temperature=0.85,
        top_p=0.95,
        max_tokens=1024,
        skip_special_tokens=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def truncate_prompt(tokenizer, prompt, max_prompt_tokens=512):
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) > max_prompt_tokens:
            tokens = tokens[:max_prompt_tokens]
            prompt = tokenizer.decode(tokens, skip_special_tokens=True)
        return prompt

    prompts_all=[]
    for prompt in prompts_chunk:
        prompt = truncate_prompt(tokenizer, prompt, max_prompt_tokens=1020)
        prompts_all.append(prompt)

    print(f"[GPU {gpu_id}] ⚙️ Đang generate...")
    outputs = llm.generate(prompts_all, sampling_params)

    with open(temp_out_path, 'w', encoding='utf-8') as f:
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            item = {
                'prompt': data_chunk[i],
                'generated_text': generated_text,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[GPU {gpu_id}] ✅ Xong! Đã lưu kết quả tạm vào {temp_out_path}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()


def run_parallel_inference(model_path, output_dir, output_file, num_gpus):
    print(f"\n{'='*60}")
    print(f"🚀 BẮT ĐẦU CHẠY MODEL (DATA PARALLEL): {model_path}")
    print(f"📂 Thư mục lưu: {os.path.join(output_dir, output_file)}")
    print(f"⚙️  Số lượng GPU sử dụng: {num_gpus}")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)

    print("⏳ Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # data = load_dataset('VoCuc/MetaMathQA-50k-256', split='train')['query']
    data = load_dataset('Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen', split='train')['instruction']

    print("🧩 Applying chat template...")
    prompts_all = []
    for prompt in data:
        # conversation = [
        #     {"role": "system", "content": "You are a teacher. Solve the problem and put your final answer within \\boxed{}."},
        #     {"role": "user", "content": prompt}
        # ]
        conversation = [
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        prompts_all.append(formatted_prompt)

    total_data = len(prompts_all)
    chunk_size = math.ceil(total_data / num_gpus)
    
    processes = []
    temp_files = []

    print(f"🔄 Chia {total_data} prompts thành {num_gpus} phần (khoảng {chunk_size} prompts/GPU)...")

    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = [x.strip() for x in cuda_devices.split(",") if x.strip()] if cuda_devices else []
    num_gpus = len(gpu_ids) if gpu_ids else torch.cuda.device_count()

    for idx, gpu_id in enumerate(gpu_ids or range(num_gpus)):
        start_idx = idx * chunk_size
        end_idx = min((idx + 1) * chunk_size, total_data)

        if start_idx >= total_data:
            break

        data_chunk = data[start_idx:end_idx]
        prompts_chunk = prompts_all[start_idx:end_idx]

        temp_out_path = os.path.join(output_dir, f"temp_worker_{idx}.jsonl")
        temp_files.append(temp_out_path)

        p = mp.Process(
            target=worker_inference,
            args=(gpu_id, model_path, data_chunk, prompts_chunk, temp_out_path)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n🔗 Tất cả workers đã xong. Đang gộp kết quả...")
    final_output_path = os.path.join(output_dir, output_file)
    
    with open(final_output_path, 'w', encoding='utf-8') as outfile:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
                os.remove(temp_file)

    print(f"🎉 HOÀN TẤT! File tổng hợp được lưu tại: {final_output_path}\n")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Run vLLM Inference with Data Parallelism")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path or HF ID of the model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output file")
    parser.add_argument("--output_file", type=str, default="generated_train.jsonl", help="Output file name")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to distribute the workload")
    
    args = parser.parse_args()
    
    run_parallel_inference(
        model_path=args.model_path,
        output_dir=args.output_dir,
        output_file=args.output_file,
        num_gpus=args.num_gpus
    )