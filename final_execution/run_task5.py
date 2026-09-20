#!/usr/bin/env python3
"""Run Task 5 workers in two eight-GPU waves, then aggregate outputs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final_execution" / "task5"
STATUS = ROOT / "artifacts" / "final_execution" / "status.json"
PYTHON = ROOT / ".venvs" / "train" / "bin" / "python"
SCRIPT = ROOT / "final_execution" / "intrinsic_analysis.py"


def write_status(payload: dict) -> None:
    payload = dict(payload, updated=time.time(), gpus=list(range(8)))
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(STATUS)


def main() -> None:
    subprocess.run([str(PYTHON), str(SCRIPT), "prepare"], cwd=ROOT, check=True)
    jobs = json.loads((OUT / "jobs.json").read_text())
    failures = []
    for wave_index in range(0, len(jobs), 8):
        wave = jobs[wave_index:wave_index + 8]
        processes = []
        for gpu, job in enumerate(wave):
            job_dir = OUT / "workers" / job["id"]
            job_dir.mkdir(parents=True, exist_ok=True)
            spec_path = job_dir / "job.json"
            spec_path.write_text(json.dumps(job, indent=2) + "\n")
            log = (job_dir / "worker.log").open("a")
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
            proc = subprocess.Popen([str(PYTHON), str(SCRIPT), "worker", "--job-json", str(spec_path)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((job, gpu, proc, log))
        while processes:
            active=[]
            for job,gpu,proc,log in processes:
                rc=proc.poll()
                if rc is None:
                    active.append((job,gpu,proc,log))
                else:
                    log.close()
                    if rc: failures.append({"job":job,"gpu":gpu,"returncode":rc})
            write_status({"phase":"task5_intrinsic","wave":wave_index//8+1,"active":[{"id":j["id"],"gpu":g,"pid":p.pid} for j,g,p,_ in active],"failures":failures})
            if failures:
                for _,_,proc,log in active:
                    proc.terminate(); log.close()
                raise RuntimeError(f"Task 5 worker failures: {failures}")
            processes=active
            if processes: time.sleep(10)
    write_status({"phase":"task5_aggregate","active":[],"failures":[]})
    subprocess.run([str(PYTHON), str(SCRIPT), "aggregate"], cwd=ROOT, check=True)
    write_status({"phase":"task5_complete","active":[],"failures":[],"task5_output":str(OUT)})


if __name__ == "__main__":
    main()
