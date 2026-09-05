#!/usr/bin/env python3
"""Create and verify an exact, ordered training-dataset contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = 1


def normalized_record(row: dict, path: Path, line_number: int) -> dict[str, str]:
    prompt = row.get("instruction", row.get("prompt"))
    output = row.get("generated_text", row.get("output"))
    if isinstance(output, list):
        output = output[0] if output else ""
    if not isinstance(prompt, str) or not isinstance(output, str):
        raise ValueError(
            f"{path}:{line_number}: expected string prompt/instruction and "
            "generated_text/output"
        )
    return {"prompt": prompt, "output": output}


def digest_jsonl(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    ordered = hashlib.sha256()
    item_hashes: list[str] = []
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = normalized_record(json.loads(line), path, line_number)
            canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            item_hash = hashlib.sha256(canonical).hexdigest()
            ordered.update(bytes.fromhex(item_hash))
            item_hashes.append(item_hash)
            count += 1
    if not count:
        raise ValueError(f"empty dataset: {path}")
    unordered = hashlib.sha256()
    for item_hash in sorted(item_hashes):
        unordered.update(bytes.fromhex(item_hash))
    return {
        "count": count,
        "ordered_sha256": ordered.hexdigest(),
        "multiset_sha256": unordered.hexdigest(),
    }


def resolve_jsonl(path: Path, split: str) -> Path:
    if path.is_file():
        return path
    candidates = [path / f"{split}.jsonl"]
    candidates.extend(sorted(path.glob(f"{split}_*.jsonl")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no {split}.jsonl or {split}_*.jsonl under {path}")


def create(reference: Path, output: Path, pair: str) -> None:
    digest = digest_jsonl(reference)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "pair": pair,
        "reference": str(reference.resolve()),
        **digest,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[created] {output}: pair={pair} count={digest['count']:,} ordered={digest['ordered_sha256']}")


def export_minillm(reference: Path, output: Path, split: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{split}.jsonl"
    count = 0
    with reference.open(encoding="utf-8") as source, target.open("w", encoding="utf-8") as sink:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = normalized_record(json.loads(line), reference, line_number)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    if not count:
        raise ValueError(f"empty dataset: {reference}")
    print(f"[exported] {target}: {count:,} ordered records")


def check(manifest: Path, candidate: Path, split: str) -> None:
    expected = json.loads(manifest.read_text())
    if expected.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported contract schema in {manifest}")
    jsonl = resolve_jsonl(candidate, split)
    actual = digest_jsonl(jsonl)
    if actual["multiset_sha256"] != expected["multiset_sha256"]:
        raise SystemExit(
            f"DATASET MISMATCH: {jsonl}\n"
            f"expected count/set: {expected['count']} {expected['multiset_sha256']}\n"
            f"actual count/set:   {actual['count']} {actual['multiset_sha256']}"
        )
    if actual["ordered_sha256"] != expected["ordered_sha256"]:
        raise SystemExit(
            f"DATASET ORDER MISMATCH: {jsonl}\n"
            "Samples are identical as a multiset but occur in a different order. "
            "Regenerate processed data with the ordered preprocessor."
        )
    print(
        f"[exact] {jsonl}: count={actual['count']:,} "
        f"multiset_sha256={actual['multiset_sha256']} "
        f"ordered_sha256={actual['ordered_sha256']}"
    )


def validate(candidate: Path, expected_count: int | None) -> None:
    jsonl = resolve_jsonl(candidate, "train")
    digest = digest_jsonl(jsonl)
    if expected_count is not None and digest["count"] != expected_count:
        raise SystemExit(
            f"DATASET COUNT MISMATCH: {jsonl}: expected {expected_count}, "
            f"got {digest['count']}"
        )
    print(f"[valid] {jsonl}: count={digest['count']:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("create")
    make.add_argument("--reference", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--pair", choices=("qwen", "qwen3"), required=True)
    verify = sub.add_parser("check")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--candidate", type=Path, required=True)
    verify.add_argument("--split", default="train")
    export = sub.add_parser("export-minillm")
    export.add_argument("--reference", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--split", choices=("train", "valid"), required=True)
    inspect = sub.add_parser("validate")
    inspect.add_argument("--candidate", type=Path, required=True)
    inspect.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    if args.command == "create":
        create(args.reference, args.output, args.pair)
    elif args.command == "check":
        check(args.manifest, args.candidate, args.split)
    elif args.command == "export-minillm":
        export_minillm(args.reference, args.output, args.split)
    else:
        validate(args.candidate, args.expected_count)


if __name__ == "__main__":
    main()
