from typing import Dict, List, Optional

from datasets import load_dataset

from .MATHLoader import MATHLoader


class MATH500Loader(MATHLoader):
    """Loader for HuggingFaceH4/MATH-500.

    Reuses the answer extraction / symbolic correctness logic from MATHLoader.
    """

    def __init__(
        self,
        base_path,
        max_samples=None,
        data_subset: str = "math500",
        balance_by_level: bool = False,
        samples_per_level: Optional[int] = None,
        fill_shortfall_to_max: bool = False,
        random_seed: int = 42,
        allowed_levels: Optional[List[int]] = None,
    ):
        super().__init__(
            base_path=base_path,
            max_samples=max_samples,
            data_subset=data_subset,
            balance_by_level=balance_by_level,
            samples_per_level=samples_per_level,
            fill_shortfall_to_max=fill_shortfall_to_max,
            random_seed=random_seed,
            allowed_levels=allowed_levels,
        )

    def load_data(self, split: str = "test") -> List[Dict[str, str]]:
        ds = load_dataset("HuggingFaceH4/MATH-500")
        if split not in ds:
            available = ", ".join(ds.keys())
            raise ValueError(f"Split '{split}' not found in HuggingFaceH4/MATH-500. Available: {available}")

        rows = ds[split]
        qa_pairs: List[Dict[str, str]] = []

        for idx, row in enumerate(rows):
            question = str(row.get("problem", "")).strip()
            solution = row.get("solution")
            final_answer = row.get("answer")
            answer = str(solution if solution not in (None, "") else final_answer).strip()
            if not question or not answer:
                continue

            level = row.get("level")
            level_int = self._parse_level(level)
            if self.allowed_levels and level_int not in set(self.allowed_levels):
                continue

            qa_pairs.append(
                {
                    "question": question,
                    "answer": answer,
                    "final_answer": None if final_answer is None else str(final_answer).strip(),
                    "subject": row.get("subject", row.get("type")),
                    "type": row.get("type", row.get("subject")),
                    "level": level,
                    "level_int": level_int,
                    "unique_id": row.get("unique_id", row.get("id")),
                    "original_idx": idx,
                }
            )

        if self.balance_by_level and self.samples_per_level:
            import random
            from collections import defaultdict

            rng = random.Random(self.random_seed)
            buckets = defaultdict(list)
            for sample in qa_pairs:
                if sample["level_int"] is None:
                    continue
                buckets[sample["level_int"]].append(sample)

            balanced: List[Dict[str, str]] = []
            for level in sorted(buckets):
                rows_for_level = buckets[level]
                rng.shuffle(rows_for_level)
                balanced.extend(rows_for_level[: self.samples_per_level])

            if self.fill_shortfall_to_max and self.max_samples and len(balanced) < self.max_samples:
                selected_keys = {
                    (row.get("unique_id"), row.get("original_idx"))
                    for row in balanced
                }
                leftovers = [
                    row for row in qa_pairs
                    if (row.get("unique_id"), row.get("original_idx")) not in selected_keys
                ]
                rng.shuffle(leftovers)
                need = max(0, int(self.max_samples) - len(balanced))
                qa_pairs = balanced + leftovers[:need]
            else:
                qa_pairs = balanced
        elif self.max_samples:
            qa_pairs = qa_pairs[: self.max_samples]

        print(f"Loaded {len(qa_pairs)} question-answer pairs from HuggingFaceH4/MATH-500 [{split}]")
        return qa_pairs
