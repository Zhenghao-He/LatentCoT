from .dataset_loader import DataLoader
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import random
from datasets import load_dataset
import re
from typing import Optional

from collections import defaultdict

class MMLULoader(DataLoader):
    """Loader for MMLU dataset."""
    def __init__(self, base_path, max_samples = None, data_subset: str = "all"):
        super().__init__(base_path, max_samples)
        self.data_subset = data_subset

    def load_data(self, split: str = 'test') -> List[Dict[str, str]]:


        ds = load_dataset("cais/mmlu", "all")
        num_subjects = len(ds[split].unique('subject'))
        uniform_test = self.sample_uniform_per_subject(ds[split], n_per_subject=self.max_samples // num_subjects + 1)

        qa_pairs = []
        for ex in uniform_test:
            q = ex["question"]
            choices = ex["choices"]
            answer = ex["answer"]
            
            letters = ["A", "B", "C", "D"]
            correct_letter = letters[answer]

            prompt = ""
            prompt += f"Question:\n{q}\n\nChoices:\n"
            for letter, text in zip(letters, choices):
                prompt += f"{letter}. {text}\n"

            prompt += "\nRespond with the letter (A, B, C, or D)."
            qa_pairs.append({"question": prompt, "answer": correct_letter})

        return qa_pairs


    def sample_uniform_per_subject(self, dataset, n_per_subject, seed=42):
        random.seed(seed)
        bucket = defaultdict(list)

        for ex in dataset:
            bucket[ex["subject"]].append(ex)

        sampled = []
        for subject, examples in bucket.items():
            k = min(n_per_subject, len(examples))
            sampled.extend(random.sample(examples, k))

        random.shuffle(sampled)
        return sampled

    

    def extract_choice_letter(self, text: str) -> Optional[str]:
        if not text:
            return None
        t = text.strip().upper()

        # 常见格式： "A" / "(A)" / "Answer: A" / "Option B" / "The answer is C"
        m = re.search(r'\b([ABCD])\b', t)
        if m:
            return m.group(1)

        # 兜底：有些会输出 "A." 或 "A)"
        m = re.search(r'^\s*([ABCD])[\.\)]', t)
        if m:
            return m.group(1)

        return None

    def check_answer_correctness(self, predicted: str, ground_truth_letter: str) -> bool:
        if not predicted or not ground_truth_letter:
            return False
        # pred_letter = self.extract_choice_letter(predicted)
        # import pdb; pdb.set_trace()
        predicted_letter = predicted.strip().upper()
        return predicted_letter == ground_truth_letter

    

    def extract_answer(self, response: str) -> str:
        """Extract final answer from response.
        
        Args:
            response: Generated response text
            
        Returns:
            Extracted answer string
        """
        # Default implementation - can be overridden by subclasses
        answer = ""
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('#'):
                answer = line
                break
        
        predicted_letter = self.extract_choice_letter(answer)
        return predicted_letter if predicted_letter else answer
    
    def extract_answer_for_hint(self, ground_truth: str) -> str:

        return ground_truth



if __name__ == "__main__":
    loader = MMLULoader(base_path="", max_samples=1000)
    data = loader.load_data(split='train')
    import pdb; pdb.set_trace()