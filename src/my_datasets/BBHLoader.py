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

class BBHLoader(DataLoader):
    """Loader for MMLU dataset."""
    def __init__(self, base_path, max_samples = None, data_subset: str = "all"):
        super().__init__(base_path, max_samples)
        self.data_subset = data_subset

    def load_data(self, split: str = 'test') -> List[Dict[str, str]]:
        data_file = self.base_path / f"{split}.json"
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_file}")
        qa_pairs = []
        with open(data_file, "r") as f:
            data = json.load(f)

        examples = data["examples"]

        for i, ex in enumerate(examples):
            i+= 1
            if self.max_samples and i > self.max_samples:
                break
            question = ex["input"]
            answer = ex["target"]

            # question += "\n\nPlease respond with the letter (A, B, C, D, etc.) corresponding to your choice."

            answer = self.extract_choice_letter(answer)

            qa_pairs.append({"question": question, "answer": answer})
        

        print(f"Loaded {len(qa_pairs)} question-answer pairs from {data_file}")
        return qa_pairs


    # def sample_uniform_per_subject(self, dataset, n_per_subject, seed=42):
    #     random.seed(seed)
    #     bucket = defaultdict(list)

    #     for ex in dataset:
    #         bucket[ex["subject"]].append(ex)

    #     sampled = []
    #     for subject, examples in bucket.items():
    #         k = min(n_per_subject, len(examples))
    #         sampled.extend(random.sample(examples, k))

    #     random.shuffle(sampled)
    #     return sampled

    

    def extract_choice_letter(self, text: str) -> Optional[str]:
        if not text:
            return None
        t = text.strip().upper()

        # 常见格式： "A" / "(A)" / "Answer: A" / "Option B" / "The answer is C"
        m = re.search(r'\b([ABCDEFGHIJKLMNOPQR])\b', t)
        if m:
            return m.group(1)

        # 兜底：有些会输出 "A." 或 "A)"
        m = re.search(r'^\s*([ABCDEFGHIJKLMNOPQR])[\.\)]', t)
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

    # def check_answer_correctness(self, predicted: str, ground_truth: str) -> bool:
    #     if not predicted or not ground_truth:
    #         return False
        
    #     # 1. 基础清理
    #     pred_clean = predicted.strip().lower()
    #     truth_clean = ground_truth.strip().lower()
        
    #     # 2. 如果完全一致直接返回
    #     if pred_clean == truth_clean:
    #         return True
        
    #     import re

    #     def extract_last_num(text):
    #         # 移除逗号，处理 "70,000" -> "70000"
    #         text = text.replace(',', '')
    #         # 寻找数字（包括正负号和小数点）
    #         nums = re.findall(r'-?\d+\.?\d*', text)
    #         if nums:
    #             try:
    #                 # 转化为 float 以处理 "70000" == "70000.0" 的情况
    #                 return float(nums[-1])
    #             except ValueError:
    #                 return None
    #         return None

    #     pred_val = extract_last_num(pred_clean)
    #     truth_val = extract_last_num(truth_clean)

    #     # 3. 数值比较
    #     if pred_val is not None and truth_val is not None:
    #         # 使用 round 或 math.isclose 处理浮点数微小误差（可选）
    #         return pred_val == truth_val

    #     return False

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
        # lines = response.strip().split('\n')
        # for line in reversed(lines):
        #     line = line.strip()
        #     if line and not line.startswith('#'):
        #         return line
        # return response.strip()
    
    def extract_answer_for_hint(self, ground_truth: str) -> str:

        return ground_truth



if __name__ == "__main__":
    loader = BBHLoader(base_path="./data/BIG-Bench-Hard/bbh/", max_samples=1000)
    data = loader.load_data(split='logical_deduction_five_objects')
    import pdb; pdb.set_trace()