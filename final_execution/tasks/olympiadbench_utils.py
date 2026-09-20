"""lm-eval adapter for the text-only English competition subset."""

from __future__ import annotations

from final_execution.math_verifier import verify_answers


def doc_to_target(doc: dict) -> str:
    answers = [str(value) for value in doc.get("final_answer", []) if str(value).strip()]
    return " || ".join(answers)


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    prediction = results[0] if results else ""
    references = [str(value) for value in doc.get("final_answer", []) if str(value).strip()]
    correct = any(verify_answers(reference, prediction)["correct"] for reference in references)
    return {"exact_match": int(correct)}
