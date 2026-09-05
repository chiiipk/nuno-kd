import math
import json
import tempfile
import unittest
from pathlib import Path

from baselines.dataset_contract import check, create, export_minillm
from baselines.eval_lm_harness import find_metric
from baselines.report_mean_std import summarize


class BaselineReportingTests(unittest.TestCase):
    def test_dataset_contract_requires_content_and_order(self):
        rows = [
            {"prompt": "p1", "generated_text": "a1"},
            {"prompt": "p2", "generated_text": "a2"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.jsonl"
            raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest = root / "manifest.json"
            create(raw, manifest, "qwen")
            processed = root / "processed"
            processed.mkdir()
            (processed / "train.jsonl").write_text(
                "".join(json.dumps({"instruction": row["prompt"], "output": row["generated_text"]}) + "\n" for row in rows)
            )
            check(manifest, processed, "train")
            export_minillm(raw, root / "mini", "train")
            check(manifest, root / "mini", "train")

            (processed / "train.jsonl").write_text(
                "".join(json.dumps({"instruction": row["prompt"], "output": row["generated_text"]}) + "\n" for row in reversed(rows))
            )
            with self.assertRaises(SystemExit):
                check(manifest, processed, "train")

    def test_extracts_group_metric_and_stderr(self):
        payload = {
            "groups": {
                "mmlu_stem": {
                    "acc,none": 0.515,
                    "acc_stderr,none": 0.012,
                }
            }
        }
        metric, value, stderr = find_metric(payload, "mmlu_stem")
        self.assertEqual(metric, "acc,none")
        self.assertAlmostEqual(value, 0.515)
        self.assertAlmostEqual(stderr, 0.012)

    def test_falls_back_to_macro_average_for_group_subtasks(self):
        payload = {
            "results": {
                "bbh_cot_fewshot_a": {"exact_match": 0.4},
                "bbh_cot_fewshot_b": {"exact_match": 0.8},
            }
        }
        _, value, stderr = find_metric(payload, "bbh_cot_fewshot")
        self.assertAlmostEqual(value, 0.6)
        self.assertIsNone(stderr)

    def test_two_seed_sample_standard_deviation(self):
        mean, std = summarize([40.0, 44.0])
        self.assertEqual(mean, 42.0)
        self.assertAlmostEqual(std, math.sqrt(8.0))

    def test_fixed_checkpoint_has_zero_std(self):
        self.assertEqual(summarize([57.1]), (57.1, 0.0))


if __name__ == "__main__":
    unittest.main()
