# Gemma-3 AMiD Evaluation Sharding

## Objective

Finish the two required Gemma-3 AMiD evaluation seeds faster by dedicating
GPUs 0–6 on `H200_ANNP36` to independent benchmark shards. Stop the active
Teacher, Student, and Qwen3 jobs without deleting their fsynced predictions.

## Frozen scientific protocol

- Checkpoint: `checkpoints/experimental_2026/gemma3/amid/seed10/checkpoint-1000`
- Evaluation seeds: 42 and 43
- Tasks: GSM8K, GSM+, MATH500, MMLU-Pro-Math, and OlympiadBench
- Greedy decoding with temperature 0
- `max_new_tokens=5120`, `max_model_len=8192`, batch size 16
- Existing pinned datasets, prompt templates, answer parser, and verifier
- Existing Gemma-3-1B tokenizer snapshot

Only scheduling changes. Model weights, examples, decoding, scoring, and seed
values remain unchanged.

## Process transition

1. Resolve the exact process groups for the active campaign.
2. Gracefully stop Teacher seed43, Student seed43, all Qwen3 queues, and the
   two monolithic AMiD evaluators. Never signal GPU7 or unrelated processes.
3. Preserve every manifest, log, and fsynced prediction JSONL. Record the
   interruption in durable audit metadata.
4. Copy already generated AMiD predictions into isolated per-task shard
   outputs. Validate IDs before resuming.

## GPU allocation

Seven durable workers use GPUs 0–6:

| GPU | Work |
|---:|---|
| 0 | seed42 GSM+ resume |
| 1 | seed43 GSM+ resume |
| 2 | seed42 MATH500 |
| 3 | seed42 MMLU-Pro-Math |
| 4 | seed42 OlympiadBench |
| 5 | seed43 MATH500, then seed43 OlympiadBench |
| 6 | seed43 MMLU-Pro-Math |

GSM8K is already complete for both seeds and is not regenerated.

## Merge and validation

After all shard workers exit zero:

1. Require the exact expected problem IDs and counts for every shard.
2. Reject duplicate or unexpected IDs.
3. Copy completed shard predictions into the original canonical AMiD seed
   directories while preserving the interrupted versions for audit.
4. Run the original evaluator in score-only mode; because every expected
   prediction is present, it must not initialize vLLM.
5. Require 14,396 unique predictions and a parseable `summary.json` for each
   seed, then compute the two-seed mean Avg-5.

## Failure handling

- Each shard has immutable manifest, PID, log, and exit marker.
- Resume only when its fingerprint matches.
- Do not change batch size or generation limits after failure.
- Do not delete partial artifacts or install packages globally.
- If a shard fails twice for the same cause, stop and report instead of
  changing the scientific protocol.
