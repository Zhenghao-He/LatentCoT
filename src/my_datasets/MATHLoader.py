from .dataset_loader import DataLoader
from collections import defaultdict
from typing import List, Dict, Optional
from datasets import load_dataset
import random
import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)


class MATHLoader(DataLoader):
    """Loader for the Competition MATH dataset."""

    def __init__(
        self,
        base_path,
        max_samples=None,
        data_subset: str = "math",
        balance_by_level: bool = False,
        samples_per_level: Optional[int] = None,
        fill_shortfall_to_max: bool = False,
        random_seed: int = 42,
        allowed_levels: Optional[List[int]] = None,
    ):
        super().__init__(base_path, max_samples)
        self.data_subset = data_subset
        self.balance_by_level = balance_by_level
        self.samples_per_level = samples_per_level
        self.fill_shortfall_to_max = fill_shortfall_to_max
        self.random_seed = random_seed
        self.allowed_levels = allowed_levels
        self._sympy_transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )

    def _parse_level(self, level: Optional[str]) -> Optional[int]:
        if not level:
            return None
        match = re.search(r"(\d+)", str(level))
        return int(match.group(1)) if match else None

    def load_data(self, split: str = "train") -> List[Dict[str, str]]:
        ds = load_dataset("qwedsacf/competition_math")
        if split not in ds:
            available = ", ".join(ds.keys())
            raise ValueError(f"Split '{split}' not found in competition_math. Available: {available}")

        rows = ds[split]
        qa_pairs: List[Dict[str, str]] = []

        for idx, row in enumerate(rows):
            question = row["problem"].strip()
            answer = row["solution"].strip()
            level = row.get("level")
            level_int = self._parse_level(level)
            if self.allowed_levels and level_int not in set(self.allowed_levels):
                continue
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "type": row.get("type"),
                "level": level,
                "level_int": level_int,
                "original_idx": idx,
            })

        if self.balance_by_level and self.samples_per_level:
            rng = random.Random(self.random_seed)
            buckets: Dict[int, List[Dict[str, str]]] = defaultdict(list)
            for row in qa_pairs:
                if row["level_int"] is None:
                    continue
                buckets[row["level_int"]].append(row)

            balanced: List[Dict[str, str]] = []
            for level in sorted(buckets):
                rows_for_level = buckets[level]
                rng.shuffle(rows_for_level)
                balanced.extend(rows_for_level[: self.samples_per_level])
            if self.fill_shortfall_to_max and self.max_samples and len(balanced) < self.max_samples:
                # Keep the balanced prefix deterministic, then top up from the leftover pool.
                selected_keys = {
                    (row.get("original_idx"), row.get("level_int"), row.get("type"))
                    for row in balanced
                }
                leftovers: List[Dict[str, str]] = []
                for row in qa_pairs:
                    key = (row.get("original_idx"), row.get("level_int"), row.get("type"))
                    if key not in selected_keys:
                        leftovers.append(row)
                rng.shuffle(leftovers)
                need = max(0, int(self.max_samples) - len(balanced))
                qa_pairs = balanced + leftovers[:need]
            else:
                qa_pairs = balanced
        elif self.max_samples:
            qa_pairs = qa_pairs[: self.max_samples]

        print(f"Loaded {len(qa_pairs)} question-answer pairs from qwedsacf/competition_math [{split}]")
        return qa_pairs

    def _extract_boxed_answer(self, text: str) -> Optional[str]:
        if not text:
            return None

        for marker in ("\\boxed{", "\\fbox{"):
            start = text.rfind(marker)
            if start == -1:
                continue

            i = start + len(marker)
            depth = 1
            chars = []
            while i < len(text):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return "".join(chars).strip()
                chars.append(ch)
                i += 1

        return None

    def _normalize_answer(self, text: str) -> str:
        if not text:
            return ""

        answer = text.strip()
        answer = answer.replace("\n", " ")
        answer = answer.replace("\\left", "").replace("\\right", "")
        answer = answer.replace("\\,", "").replace("\\!", "")
        answer = re.sub(r"\$+", "", answer)
        answer = re.sub(r"\s+", "", answer)
        answer = answer.rstrip(".")
        return answer

    def _latex_to_sympy(self, text: str) -> str:
        expr = text.strip()
        expr = expr.replace("\n", " ")
        expr = expr.replace("\\left", "").replace("\\right", "")
        expr = expr.replace("\\cdot", "*").replace("\\times", "*")
        expr = expr.replace("\\pi", "pi")
        expr = expr.replace("^", "**")
        expr = expr.replace("{", "(").replace("}", ")")

        while "\\frac" in expr:
            expr = re.sub(r"\\frac\s*\(([^()]+)\)\s*\(([^()]+)\)", r"((\1)/(\2))", expr)
            expr = re.sub(r"\\frac\s*([^\s()]+)\s*([^\s()]+)", r"((\1)/(\2))", expr)
            if "\\frac" in expr:
                break

        expr = re.sub(r"\\sqrt\s*\(([^()]+)\)", r"sqrt(\1)", expr)
        expr = re.sub(r"\\sqrt\s*([^\s()]+)", r"sqrt(\1)", expr)
        expr = expr.replace("$", "")
        expr = re.sub(r"\\text\([^)]*\)", "", expr)
        expr = re.sub(r"\\[a-zA-Z]+", "", expr)
        expr = re.sub(r"\s+", "", expr)
        return expr

    def _to_sympy_expr(self, text: str) -> Optional[sp.Expr]:
        if not text:
            return None

        candidate = self._extract_boxed_answer(text)
        if candidate is None:
            candidate = self._extract_last_math_span(text)
        if candidate is None:
            candidate = text.strip().split("\n")[-1].strip()
        if "=" in candidate:
            candidate = candidate.split("=")[-1].strip()

        normalized = self._latex_to_sympy(candidate)
        if not normalized:
            return None

        try:
            return parse_expr(
                normalized,
                transformations=self._sympy_transformations,
                evaluate=True,
            )
        except Exception:
            try:
                return sp.sympify(normalized)
            except Exception:
                return None

    def _extract_last_math_span(self, text: str) -> Optional[str]:
        matches = re.findall(r"\$([^$]+)\$", text)
        if matches:
            return matches[-1].strip()
        return None

    def extract_answer(self, response: str) -> str:
        boxed = self._extract_boxed_answer(response)
        if boxed is not None:
            return boxed

        math_span = self._extract_last_math_span(response)
        if math_span is not None:
            return math_span

        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        return lines[-1] if lines else ""

    def extract_answer_for_hint(self, ground_truth: str) -> str:
        boxed = self._extract_boxed_answer(ground_truth)
        if boxed is not None:
            return boxed

        math_span = self._extract_last_math_span(ground_truth)
        if math_span is not None:
            return math_span

        lines = [line.strip() for line in ground_truth.strip().split("\n") if line.strip()]
        return lines[-1] if lines else ground_truth.strip()

    def check_answer_correctness(self, predicted: str, ground_truth: str) -> bool:
        if not predicted or not ground_truth:
            return False

        pred_norm = self._normalize_answer(predicted)
        truth_norm = self._normalize_answer(self.extract_answer_for_hint(ground_truth))
        if pred_norm == truth_norm:
            return True

        pred_expr = self._to_sympy_expr(predicted)
        truth_expr = self._to_sympy_expr(ground_truth)
        if pred_expr is not None and truth_expr is not None:
            try:
                return bool(sp.simplify(pred_expr - truth_expr) == 0)
            except Exception:
                try:
                    return bool(pred_expr.equals(truth_expr))
                except Exception:
                    return False

        pred_num = re.findall(r"-?\d+\.?\d*", pred_norm)
        truth_num = re.findall(r"-?\d+\.?\d*", truth_norm)
        if pred_num and truth_num:
            return pred_num[-1] == truth_num[-1]

        return False


if __name__ == "__main__":
    loader = MATHLoader(base_path="", max_samples=5)
    data = loader.load_data(split="train")
    print(data[0])
