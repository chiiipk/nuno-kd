# CST ablations on 6 x H200

The default study uses Qwen, full exact-matched data, seeds 10 and 42, the
same output-KD objective and optimizer as the main CST run, and all eight
lm-eval tasks. The main configuration is `q=2`, gamma range `[1e-2,1e2]`,
log-uniform random sampling, four supervised layers, and 64 response tokens.

## Tables

1. `objective`: Hidden MSE, token-Gram MSE, linear CKA, full normalized
   eigen-spectrum L2, direct unnormalized spectrum Smooth-L1, and CST.
2. `num_gamma`: `q` in `{1,2,4,8}`.
3. `gamma`: random ranges `[1e-2,1]`, `[1e-1,1e1]`, `[1e-2,1e2]`,
   `[1,1e2]`, plus a fixed endpoint grid on `[1e-2,1e2]`.
4. `layers`: supervised layer count in `{1,2,4}`.
5. `tokens`: token count in `{32,64,128}`, including performance and training
   timing in the raw logs.

Identical main configurations are trained only once as `cst_main` and reused
across tables. Hidden MSE explicitly learns a student-to-teacher linear
projector because widths differ. Gram, CKA, spectrum baselines, and CST are
projector-free. Eigendecomposition is used only by the two spectrum baselines,
never by CST.

The auxiliary weight defaults to `0.003`. Because objective scales differ, the
objective table is a fixed-weight controlled comparison, not a claim that each
baseline has been optimally tuned. Override `AUX_WEIGHT` only if applying the
same declared tuning protocol to every objective.

## Run

Create and verify the dataset contracts first, then:

```bash
chmod +x ablations/run_6xh200.sh
bash ablations/run_6xh200.sh train
bash ablations/run_6xh200.sh eval
bash ablations/run_6xh200.sh report
```

Use `PAIR=gemma` for a separate Gemma study; Qwen is the default. Results are
resumable under `results/ablations/`, raw lm-eval samples under
`benchmark_results/ablations/`, and per-table CSV/LaTeX/JSON under
`benchmark_results/ablations/tables/`.

The table reports mean plus sample standard deviation across the two training
seeds. lm-eval per-example stderr remains in each `scores.json` and is not used
as seed variance.
