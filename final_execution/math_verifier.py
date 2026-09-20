"""Versioned answer extraction and verification for generative math tasks."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Iterable


VERIFIER_VERSION = "unified-math-v2"
DEFAULT_ABS_TOL = Decimal("1e-6")
DEFAULT_REL_TOL = Decimal("1e-6")


@dataclass(frozen=True)
class Extraction:
    candidates: tuple[str, ...]
    failure_type: str | None = None


def _balanced_braced(text: str, open_index: int) -> tuple[str | None, int]:
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : index], index + 1
    return None, len(text)


def _boxed_candidates(text: str) -> tuple[list[str], bool]:
    results: list[str] = []
    unclosed = False
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        value, _ = _balanced_braced(text, match.end() - 1)
        if value is None:
            unclosed = True
        elif value.strip():
            results.append(value.strip())
    return results, unclosed


def _first_nonempty_line(value: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return lines[0] if lines else value.strip()


def _anchored_candidate(value: str) -> str:
    """Extract one answer following an explicit answer label."""
    remainder = value.strip()
    for pattern in (
        r"(?s)^\$\$\s*(.+?)\s*\$\$",
        r"(?s)^\\\[\s*(.+?)\s*\\\]",
    ):
        match = re.match(pattern, remainder)
        if match:
            return match.group(1).strip()
    return _first_nonempty_line(remainder)


def extract_candidates(text: object, *, reference: bool = False) -> Extraction:
    if text is None:
        return Extraction((), "missing_generation")
    value = str(text).strip()
    if not value:
        return Extraction((), "missing_generation")

    boxed, unclosed = _boxed_candidates(value)
    if boxed:
        # The final box is authoritative. Earlier boxes are usually intermediate
        # values and accepting any of them creates false positives.
        return Extraction((boxed[-1],))

    if reference:
        return Extraction((value,))

    for pattern in (
        r"(?is)final\s+answer\s*(?:is|:|=)\s*(.+)",
        r"(?is)answer\s*(?:is|:|=)\s*(.+)",
        r"(?m)^\s*####\s*(.+)$",
    ):
        matches = list(re.finditer(pattern, value))
        if matches:
            candidate = _anchored_candidate(matches[-1].group(1))
            if candidate:
                return Extraction((candidate,))

    if unclosed:
        return Extraction((), "unclosed_box")

    # Unlabelled fallbacks are accepted only from the terminal line. Searching
    # the whole rationale can mistake a number from the problem statement or an
    # intermediate equation for the answer.
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    terminal = lines[-1] if lines else ""
    if len(terminal) <= 240:
        latex = re.fullmatch(r"\s*\${1,2}(.+?)\${1,2}[.!]?\s*", terminal, flags=re.S)
        if latex:
            return Extraction((latex.group(1).strip(),))
        numeric = re.fullmatch(
            r"[*_`\s]*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?%?)[.*_`\s]*",
            terminal,
        )
        if numeric:
            return Extraction((numeric.group(1).strip(),))
        conclusions = list(re.finditer(
            r"(?is)(?:^|[.!?]\s+)(?:therefore|thus|hence|so|consequently|we\s+conclude|"
            r"the\s+(?:answer|value|result|solution|minimum|maximum|number|range|probability|area|year))\b",
            terminal,
        ))
        if conclusions:
            conclusion = terminal[conclusions[-1].start():].lstrip(".!? ")
            tails = list(re.finditer(r"(?is)\b(?:is|are|equals?)\s+(.+)$", conclusion))
            if tails:
                return Extraction((_first_nonempty_line(tails[-1].group(1)),))
            equations = list(re.finditer(r"(?is)([^,;]{1,80}=.+)$", conclusion))
            if equations:
                return Extraction((_first_nonempty_line(equations[-1].group(1)),))
            inline_math = re.findall(
                r"(?<!\\)\$((?:\\\$|[^$])+?)(?<!\\)\$",
                conclusion,
                flags=re.S,
            )
            if inline_math:
                return Extraction((inline_math[-1].strip(),))
            trailing_numbers = re.findall(
                r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?%?",
                conclusion,
            )
            if trailing_numbers:
                return Extraction((trailing_numbers[-1].strip(),))
    return Extraction((), "no_candidate")


def _strip_simple_assignment(candidate: str) -> str:
    value = candidate.strip()
    if value.count(r"\in") == 1:
        left, right = value.split(r"\in", 1)
        if re.search(r"[A-Za-z\\]", left) and right.strip():
            value = right.strip()
    if value.count("=") != 1 or any(token in value for token in ("<=", ">=", "\\le", "\\ge")):
        return value
    left, right = value.split("=", 1)
    if re.search(r"[A-Za-z\\]", left) and right.strip():
        return right.strip()
    return value


def normalize(candidate: str) -> str:
    value = _strip_simple_assignment(candidate).strip()
    value = value.replace("−", "-").replace("–", "-")
    value = value.replace(r"\$", "")
    value = value.replace(r"\left", "").replace(r"\right", "")
    for token in (r"\,", r"\!", r"\quad", r"\;", r"\:"):
        value = value.replace(token, "")
    value = re.sub(r"^\$+|\$+$", "", value.strip())
    value = re.sub(r"\\(?:text|mathrm|operatorname)\s*\{([^{}]*)\}", r"\1", value)
    value = value.replace("°", "").replace(r"^\circ", "").replace(r"\circ", "")
    value = value.replace(r"\%", "%")
    value = value.strip().strip("`*_ ").rstrip(".。;")
    value = re.sub(r"\s+", "", value)
    return value


def _without_thousands_commas(value: str) -> str | None:
    if "," not in value:
        return value
    if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?", value):
        return value.replace(",", "")
    return None


def _numeric(candidate: str) -> tuple[Decimal, bool] | None:
    value = normalize(candidate).replace("$", "")
    value = _without_thousands_commas(value)
    if value is None:
        return None
    percent = value.endswith("%")
    if percent:
        value = value[:-1]
    latex_fraction = re.fullmatch(r"\\(?:d)?frac\{([-+]?\d+(?:\.\d+)?)\}\{([-+]?\d+(?:\.\d+)?)\}", value)
    try:
        if latex_fraction:
            result = Decimal(latex_fraction.group(1)) / Decimal(latex_fraction.group(2))
        elif re.fullmatch(r"[-+]?\d+(?:\.\d+)?/[-+]?\d+(?:\.\d+)?", value):
            numerator, denominator = value.split("/", 1)
            result = Decimal(numerator) / Decimal(denominator)
        elif re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?", value):
            result = Decimal(value)
        else:
            return None
    except (InvalidOperation, ZeroDivisionError):
        return None
    return result, percent


def _close(left: Decimal, right: Decimal) -> bool:
    gap = abs(left - right)
    scale = max(abs(left), abs(right), Decimal(1))
    return gap <= max(DEFAULT_ABS_TOL, DEFAULT_REL_TOL * scale)


def _numeric_equal(left: tuple[Decimal, bool], right: tuple[Decimal, bool]) -> bool:
    lvalue, lpercent = left
    rvalue, rpercent = right
    if _close(lvalue, rvalue):
        return True
    lscaled = lvalue / Decimal(100) if lpercent else lvalue
    rscaled = rvalue / Decimal(100) if rpercent else rvalue
    return _close(lscaled, rscaled)


def _pairs(reference: Iterable[str], prediction: Iterable[str]):
    for ref in reference:
        for pred in prediction:
            yield ref, pred


def _is_interval_or_union(candidate: str) -> bool:
    value = normalize(candidate)
    return any(token in value for token in (r"\cup", r"\cap", "∪", "∩"))


def _is_structured(candidate: str) -> bool:
    value = normalize(candidate)
    without_thousands = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value)
    return _is_interval_or_union(value) or "," in without_thousands


def _number_sequence(candidate: str) -> list[Decimal]:
    values = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", candidate)
    output = []
    for value in values:
        parsed = _numeric(value)
        if parsed is not None:
            output.append(parsed[0])
    return output


def _structured_numeric_equal(reference: str, prediction: str) -> bool:
    if _is_interval_or_union(reference) or _is_interval_or_union(prediction):
        return False
    left_normalized, right_normalized = normalize(reference), normalize(prediction)
    bracketed = re.compile(r"^[\[(].*,.*[\])]$")
    if bracketed.fullmatch(left_normalized) and bracketed.fullmatch(right_normalized):
        return False
    left = _number_sequence(reference)
    right = _number_sequence(prediction)
    return bool(left) and len(left) == len(right) and all(_close(x, y) for x, y in zip(left, right))


def verify_answers(reference_text: object, prediction_text: object) -> dict:
    reference = extract_candidates(reference_text, reference=True)
    prediction = extract_candidates(prediction_text)
    result = {
        "extracted_reference": list(reference.candidates),
        "extracted_prediction": list(prediction.candidates),
        "normalized_reference": [normalize(x) for x in reference.candidates],
        "normalized_prediction": [normalize(x) for x in prediction.candidates],
        "correct": False,
        "verification_method": None,
        "failure_type": None,
        "verifier_version": VERIFIER_VERSION,
    }
    if not reference.candidates:
        result["failure_type"] = reference.failure_type or "parse_error"
        return result
    if not prediction.candidates:
        result["failure_type"] = prediction.failure_type or "no_candidate"
        return result

    for ref, pred in _pairs(reference.candidates, prediction.candidates):
        if normalize(ref).casefold() == normalize(pred).casefold():
            result.update(correct=True, verification_method="exact")
            return result

    for ref, pred in _pairs(reference.candidates, prediction.candidates):
        if _is_structured(ref) or _is_structured(pred):
            if _structured_numeric_equal(ref, pred):
                result.update(correct=True, verification_method="structured_numeric")
                return result
            continue
        left, right = _numeric(ref), _numeric(pred)
        if left is not None and right is not None and _numeric_equal(left, right):
            result.update(correct=True, verification_method="numeric")
            return result

    parse_error = False
    try:
        from math_verify import parse, verify

        for ref, pred in _pairs(reference.candidates, prediction.candidates):
            if _is_structured(ref) or _is_structured(pred) or pred.count("=") > 1:
                continue
            ref_value, pred_value = _strip_simple_assignment(ref), _strip_simple_assignment(pred)
            gold = parse(ref_value, parsing_timeout=5, raise_on_error=False)
            target = parse(pred_value, parsing_timeout=5, raise_on_error=False)
            if not gold or (
                re.search(r"[A-Za-z\\^_{}]", ref_value)
                and not any(hasattr(x, "free_symbols") and x.free_symbols for x in gold)
            ):
                gold = parse(f"${ref_value}$", parsing_timeout=5, raise_on_error=False)
            if not target or (
                re.search(r"[A-Za-z\\^_{}]", pred_value)
                and not any(hasattr(x, "free_symbols") and x.free_symbols for x in target)
            ):
                target = parse(f"${pred_value}$", parsing_timeout=5, raise_on_error=False)
            if not gold or not target:
                parse_error = True
                continue
            if verify(gold, target, timeout_seconds=5, strict=True, raise_on_error=False):
                result.update(correct=True, verification_method="symbolic")
                return result
    except TimeoutError:
        result["failure_type"] = "verification_timeout"
        return result
    except Exception:
        parse_error = True

    result["failure_type"] = "parse_error" if parse_error else "not_equivalent"
    return result
