# NuNo-KD

Knowledge distillation of a small LLM from a larger one. Every method trains the
student with the same output-level KD loss (skewed forward KL by default) and adds
one auxiliary loss on the hidden states of a few matched layers:

| `--aux-loss` | Matches |
|---|---|
| `hidden_mse` | student hidden states, through a learned projector, against the teacher's (MSE) |
| `gram` | centered, trace-normalized token Gram matrices (MSE) |
| `cka` | the same Gram matrices, by linear CKA |
| `normalized_spectrum` | the normalized eigenvalue spectra of those Gram matrices |
| `direct_spectrum` | the eigenvalue spectra of the token covariances |
| `cst` | the Characteristic Spectral Transform, logdet(I + γG) over sampled γ |

All but `hidden_mse` compare token-by-token geometry, so the student and teacher
widths may differ.

## Layout

```
nuno_kd/                 training package
  train.py               trainer entry point (torchrun -m nuno_kd.train)
  args.py                command-line arguments
  modeling.py            model/tokenizer loading, layer selection, projectors
  distributed.py         process group, seeding, rank-0 logging
  data/prepare.py        convert and tokenize the teacher-generation dataset
  data/contract.py       exact, ordered dataset check
  data/indexed.py        memory-mapped token storage
  data/dataset.py        training dataset and collation
  losses/output_kd.py    output-level KD divergences
  losses/structural.py   hidden_mse and token-geometry losses
  losses/cst.py          CST loss
evaluation/              lm-eval-harness suite, Hendrycks MATH rescoring, tables
scripts/run_experiment.sh  config-driven prepare / train / eval / report pipeline
configs/                 experiment YAML and the DeepSpeed config
run-inputs/, run-outputs/  plans and reports of the server runs
paper/                   paper sources
```

## Setup

Training environment (Python 3.10, CUDA 12.8):

```bash
uv sync
```

Evaluator, in its own venv under `evaluation/vendor/`:

```bash
LM_EVAL_REF=6d642546f4688648fced259eb3302efd36ece5af bash evaluation/setup.sh
```

## Run an experiment

Everything is driven by one YAML file; [configs/qwen25_tsd_structural.yaml](configs/qwen25_tsd_structural.yaml)
distils Qwen2.5-14B-Instruct into Qwen2.5-1.5B-Instruct on 8 GPUs.

```bash
bash scripts/run_experiment.sh check configs/qwen25_tsd_structural.yaml
```

```bash
bash scripts/run_experiment.sh all configs/qwen25_tsd_structural.yaml
```

The phases can also run one at a time (`prepare`, `train`, `eval`, `report`).
Finished training runs and evaluations are skipped when the pipeline is
restarted. Outputs:

- `results/<experiment>/<method>/seed<seed>/<step>/`: checkpoints, one per epoch, plus `log.txt` and `train.log`;
- `benchmark_results/<experiment>/<pair>/<model>/seed<seed>/scores.json`: per-task scores;
- `benchmark_results/<experiment>/tables/`: mean ± std tables (CSV, LaTeX, JSON).

## Evaluation

`evaluation/eval_lm_harness.py` runs GSM8K, GSM-Plus, MATH (minerva_math,
`math_verify`), MBPP, SciQ, MMLU-STEM, MMLU-Pro-Math and BBH-COT with vLLM, one
GPU per task. `--tasks hendrycks_math` runs Hendrycks MATH instead, and
`evaluation/score_hendrycks_math.py` regrades its samples with `\boxed{}`
exact match and `math_verify`, since lm-eval's own `exact_match` does not
extract the final answer of a worked chat solution.
