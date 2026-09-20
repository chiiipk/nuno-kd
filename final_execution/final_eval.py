#!/usr/bin/env python3
"""Resume-safe eight-GPU final evaluation runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final_execution" / "final_eval"
TASK_PATH = ROOT / "final_execution" / "tasks"
PYTHON = ROOT / ".venvs" / "evaluator" / "bin" / "python"
STATUS = ROOT / "artifacts" / "final_execution" / "status.json"
GENERATIVE_TASKS = ["gsm8k", "gsm_plus", "mmlu_pro_math", "aime24", "aime25", "hendrycks_math500", "olympiadbench"]
LIKELIHOOD_TASKS = ["mmlu_stem", "gpqa_diamond_zeroshot"]
CODE_TASKS = ["mbpp"]
ALL_TASKS = GENERATIVE_TASKS + LIKELIHOOD_TASKS + CODE_TASKS


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def best(relative: str) -> str:
    return json.loads((ROOT / relative).read_text())["checkpoint"]


def snapshot(repo_dir: str) -> str:
    values = sorted((Path(repo_dir) / "snapshots").iterdir())
    if len(values) != 1:
        raise RuntimeError(f"expected one snapshot in {repo_dir}: {values}")
    return str(values[0].resolve())


def model_manifest() -> list[dict]:
    qbase=snapshot("/home/annp36/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct")
    qteacher=snapshot("/home/annp36/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct")
    q3base=snapshot("/home/annp36/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B")
    q3teacher=snapshot("/home/annp36/.cache/huggingface/hub/models--Qwen--Qwen3-8B")
    specs = [
        ("qwen","teacher","na",qteacher),("qwen","base","na",qbase),
        ("qwen","distillm","10",best("checkpoints/main/qwen/distillm/seed10/best.json")),
        ("qwen","csd","10",best("checkpoints/main/qwen/csd/seed10/best.json")),
        ("qwen","cka","10",best("checkpoints/final/cka_repo/qwen/seed10/best.json")),
        ("qwen","cka","42",best("checkpoints/final/cka_repo/qwen/seed42/best.json")),
        ("qwen","cst_w005_l4","10",best("checkpoints/final/cst/qwen/cst_w005_l4/seed10/best.json")),
        ("qwen","cst_w005_l4","42",best("checkpoints/final/cst/qwen/cst_w005_l4/seed42/best.json")),
        ("qwen3","teacher","na",q3teacher),("qwen3","base","na",q3base),
        ("qwen3","distillm","10",best("checkpoints/main/qwen3/distillm/seed10/best.json")),
        ("qwen3","csd","10",best("checkpoints/main/qwen3/csd/seed10/best.json")),
        ("qwen3","cka","10",best("checkpoints/final/cka_repo/qwen3/seed10/best.json")),
        ("qwen3","cka","42",best("checkpoints/final/cka_repo/qwen3/seed42/best.json")),
    ]
    for config in ["cst_w005_l2","cst_w005_l4","cst_w005_l8","cst_w008_l4"]:
        for seed in ["10","42"]:
            specs.append(("qwen3",config,seed,best(f"checkpoints/final/cst/qwen3/{config}/seed{seed}/best.json")))
    tokenizers = {"qwen": qbase, "qwen3": q3base}
    return [{"id":f"{pair}-{method}-seed{seed}" if seed!="na" else f"{pair}-{method}","pair":pair,"method":method,"train_seed":seed,"checkpoint":checkpoint,"tokenizer":tokenizers[pair]} for pair,method,seed,checkpoint in specs]


def prepare(_: argparse.Namespace) -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    payload={"models":model_manifest(),"tasks":ALL_TASKS,"generative_tasks":GENERATIVE_TASKS,"likelihood_tasks":LIKELIHOOD_TASKS,"code_tasks":CODE_TASKS,"eval_seed":42,"decoding":{"do_sample":False,"temperature":0.0,"repeats":1,"max_gen_toks":5120},"lm_eval_git_commit":subprocess.check_output(["git","-C",str(ROOT/"baselines/vendor/lm-evaluation-harness"),"rev-parse","HEAD"],text=True).strip(),"task_hashes":{p.name:sha256(p) for p in sorted(TASK_PATH.iterdir()) if p.is_file()}}
    atomic_json(OUT/"manifest.json",payload)
    print(json.dumps({"models":len(payload["models"]),"tasks":len(ALL_TASKS)}))


def result_tasks(directory: Path) -> set[str]:
    tasks=set()
    for path in directory.rglob("results*.json"):
        try:
            payload=json.loads(path.read_text()); tasks.update(payload.get("results",{})); tasks.update(payload.get("groups",{}))
        except Exception:
            pass
    return tasks


def run_phase(job: dict, phase: str, tasks: list[str], output: Path, limit: int | None) -> None:
    marker=output/f"{phase}_COMPLETE"
    if marker.exists() and all(any(name==task or name.startswith(task+"_") for name in result_tasks(output/phase)) for task in tasks):
        return
    phase_dir=output/phase; phase_dir.mkdir(parents=True,exist_ok=True)
    model_args=f"pretrained={job['checkpoint']},tokenizer={job['tokenizer']},tensor_parallel_size=1,dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=8192"
    if phase=="generative" and job["pair"]=="qwen3": model_args += ",enable_thinking=False"
    # A single auto-sized likelihood batch can enqueue all ~13k MMLU/GPQA
    # requests and trigger a vLLM async-output serialization race.  Chunking
    # this phase keeps the GPUs saturated while bounding each IPC response.
    batch_size = "512" if phase == "likelihood" else "auto"
    cmd=[str(PYTHON),"-m","lm_eval","run","--model","vllm","--model_args",model_args,"--tasks",",".join(tasks),"--batch_size",batch_size,"--log_samples","--output_path",str(phase_dir),"--include_path",str(TASK_PATH),"--seed","42,42,42,42","--cache_requests","true"]
    if phase != "code": cmd += ["--apply_chat_template","--fewshot_as_multiturn"]
    if phase in {"generative","code"}: cmd += ["--gen_kwargs","max_gen_toks=5120","temperature=0.0","do_sample=False"]
    if phase=="code": cmd += ["--confirm_run_unsafe_code"]
    if limit is not None: cmd += ["--limit",str(limit)]
    with (output/f"{phase}.log").open("a") as log:
        log.write("COMMAND "+json.dumps(cmd)+"\n"); log.flush()
        completed=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,text=True)
    if completed.returncode: raise subprocess.CalledProcessError(completed.returncode,cmd)
    present=result_tasks(phase_dir)
    missing=[task for task in tasks if not any(name==task or name.startswith(task+"_") for name in present)]
    sample_files=list(phase_dir.rglob("samples*.jsonl"))
    if missing or not sample_files: raise RuntimeError(f"{job['id']} {phase}: missing={missing}, sample_files={len(sample_files)}")
    marker.write_text(json.dumps({"tasks":tasks,"results":sorted(present),"sample_files":len(sample_files),"completed_at":time.time()})+"\n")


def worker(args: argparse.Namespace) -> None:
    if args.job_json:
        job=json.loads(Path(args.job_json).read_text())
    else:
        models=json.loads((OUT/"manifest.json").read_text())["models"]
        job=next((value for value in models if value["id"]==args.job_id),None)
        if job is None: raise ValueError(f"unknown --job-id {args.job_id}")
    output=Path(args.output) if args.output else OUT/"runs"/job["id"]
    output.mkdir(parents=True,exist_ok=True)
    phases=[("generative",GENERATIVE_TASKS),("likelihood",LIKELIHOOD_TASKS),("code",CODE_TASKS)]
    for phase,tasks in phases:
        atomic_json(output/"status.json",{"job":job,"phase":phase,"tasks":tasks,"updated":time.time()})
        run_phase(job,phase,tasks,output,args.limit)
    atomic_json(output/"complete.json",{"job":job,"tasks":ALL_TASKS,"completed_at":time.time()})


def write_status(payload: dict) -> None:
    atomic_json(STATUS,dict(payload,updated=time.time(),gpus=list(range(8))))


def run_all(args: argparse.Namespace) -> None:
    if not (OUT/"manifest.json").exists(): prepare(args)
    jobs=json.loads((OUT/"manifest.json").read_text())["models"]
    pending=[job for job in jobs if not (OUT/"runs"/job["id"]/'complete.json').exists()]
    available_gpus=list(range(8))
    active=[]
    failures=[]

    while pending or active:
        while pending and available_gpus:
            gpu=available_gpus.pop(0)
            job=pending.pop(0)
            out=OUT/"runs"/job["id"]
            out.mkdir(parents=True,exist_ok=True)
            spec=out/"job.json"; atomic_json(spec,job)
            log=(out/"worker.log").open("a")
            env=os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES":str(gpu),"HF_ALLOW_CODE_EVAL":"1","PYTHONUNBUFFERED":"1","VLLM_USE_FLASHINFER_SAMPLER":"0","VLLM_WORKER_MULTIPROC_METHOD":"spawn"})
            cmd=[str(PYTHON),str(Path(__file__)),"worker","--job-json",str(spec)]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            active.append((job,gpu,proc,log))

        waiting=[]
        for job,gpu,proc,log in active:
            rc=proc.poll()
            if rc is None:
                waiting.append((job,gpu,proc,log))
            else:
                log.close()
                available_gpus.append(gpu)
                available_gpus.sort()
                if rc:
                    failures.append({"job":job,"gpu":gpu,"returncode":rc})
        active=waiting
        write_status({"phase":"final_eval","scheduler":"dynamic","active":[{"id":j["id"],"gpu":g,"pid":p.pid} for j,g,p,_ in active],"queued":len(pending),"completed":sum((OUT/"runs"/j["id"]/"complete.json").exists() for j in jobs),"total":len(jobs),"failures":failures})
        if failures:
            for _,_,proc,log in active:
                proc.terminate()
                log.close()
            raise RuntimeError(f"final eval failures: {failures}")
        if active and not (pending and available_gpus):
            time.sleep(15)
    write_status({"phase":"final_eval_inference_complete","active":[],"completed":len(jobs),"total":len(jobs),"failures":[]})
    (OUT/"INFERENCE_COMPLETE").write_text(json.dumps({"models":len(jobs),"tasks":ALL_TASKS,"completed_at":time.time()})+"\n")


def main() -> None:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare"); p.set_defaults(func=prepare)
    p=sub.add_parser("worker"); group=p.add_mutually_exclusive_group(required=True); group.add_argument("--job-json"); group.add_argument("--job-id"); p.add_argument("--output"); p.add_argument("--limit",type=int); p.set_defaults(func=worker)
    p=sub.add_parser("run"); p.set_defaults(func=run_all)
    args=parser.parse_args(); args.func(args)


if __name__=="__main__": main()
