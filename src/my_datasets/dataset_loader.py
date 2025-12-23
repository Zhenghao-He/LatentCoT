"""Simple dataset loader for experiments."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import random
from datasets import load_dataset
import re
from typing import Optional


class DataLoader:
    """Base class for dataset loading."""
    def __init__(self, base_path: str, max_samples: Optional[int] = None):
        self.base_path = Path(base_path)
        self.max_samples = max_samples

    def load_data(self, split: str = 'train') -> List[Dict[str, str]]:
        """To be implemented by subclasses."""
        raise NotImplementedError

    def _extract_qa_from_data(self, data: Dict[str, Any]) -> tuple[str, str]:
        """To be implemented by subclasses."""
        raise NotImplementedError
    
    def extract_answer(self, response: str) -> str:
        """To be implemented by subclasses."""
        raise NotImplementedError


    def check_answer_correctness(self, predicted: str, ground_truth: str) -> bool:
        """To be implemented by subclasses."""
        raise NotImplementedError


class GSM8KLoader(DataLoader):
    """Loader for GSM8K dataset."""
    def load_data(self, split: str = 'train') -> List[Dict[str, str]]:
        data_file = self.base_path / f"{split}.jsonl"
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_file}")
        qa_pairs = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                question, answer = self._extract_qa_from_data(data)
                if question:
                    qa_pairs.append({"question": question, "answer": answer})
                    if self.max_samples and len(qa_pairs) >= self.max_samples:
                        break
        print(f"Loaded {len(qa_pairs)} question-answer pairs from {data_file}")
        return qa_pairs

    def _extract_qa_from_data(self, data: Dict[str, Any]) -> tuple[str, str]:
        question = data.get('question', '')
        answer = data.get('answer', '')
        return question.strip(), answer.strip()

    # def check_answer_correctness(self, predicted: str, ground_truth: str) -> bool:
    #     if not predicted or not ground_truth:
    #         return False
    #     pred_clean = predicted.strip().lower()
    #     truth_clean = ground_truth.strip().lower()
    #     if pred_clean == truth_clean:
    #         return True
    #     import re
    #     pred_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', pred_clean)
    #     truth_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', truth_clean)
    #     if pred_numbers and truth_numbers:
    #         return pred_numbers[-1] == truth_numbers[-1]
    #     return False
    def check_answer_correctness(self, predicted: str, ground_truth: str) -> bool:
        if not predicted or not ground_truth:
            return False
        
        # 1. 基础清理
        pred_clean = predicted.strip().lower()
        truth_clean = ground_truth.strip().lower()
        
        # 2. 如果完全一致直接返回
        if pred_clean == truth_clean:
            return True
        
        import re

        def extract_last_num(text):
            # 移除逗号，处理 "70,000" -> "70000"
            text = text.replace(',', '')
            # 寻找数字（包括正负号和小数点）
            nums = re.findall(r'-?\d+\.?\d*', text)
            if nums:
                try:
                    # 转化为 float 以处理 "70000" == "70000.0" 的情况
                    return float(nums[-1])
                except ValueError:
                    return None
            return None

        pred_val = extract_last_num(pred_clean)
        truth_val = extract_last_num(truth_clean)

        # 3. 数值比较
        if pred_val is not None and truth_val is not None:
            # 使用 round 或 math.isclose 处理浮点数微小误差（可选）
            return pred_val == truth_val

        return False
    

    def extract_answer(self, response: str) -> str:
        """Extract final answer from response.
        
        Args:
            response: Generated response text
            
        Returns:
            Extracted answer string
        """
        # Default implementation - can be overridden by subclasses
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('#'):
                return line
        return response.strip()
    
    def extract_answer_for_hint(self, ground_truth: str) -> str:
        """Extract final answer from ground truth for hint strategy.
        
        Args:
            ground_truth: Ground truth answer text
            
        Returns:
            Extracted answer number as string
        """
        if '####' in ground_truth:
            # Split by #### and take the part after it
            answer_part = ground_truth.split('####')[-1].strip()
            return answer_part
        return ground_truth.strip()
    

class GPQALoader(DataLoader):
    """Loader for GPQA dataset."""
    def __init__(self, base_path, max_samples = None, data_subset: str = "gpqa_diamond"):
        super().__init__(base_path, max_samples)
        self.data_subset = data_subset

    def load_data(self, split: str = 'train') -> List[Dict[str, str]]:


        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset("Idavidrein/gpqa", self.data_subset) # train 198 samples

        # ds = load_dataset("Idavidrein/gpqa", "gpqa_experts") # 60 samples

        # ds = load_dataset("Idavidrein/gpqa", "gpqa_extended") #546 samples
        train = ds['train']
        questions = train['Pre-Revision Question']
        correct_answers = train['Pre-Revision Correct Answer']
        incorrect_answers_1 = train['Pre-Revision Incorrect Answer 1']
        incorrect_answers_2 = train['Pre-Revision Incorrect Answer 2']
        incorrect_answers_3 = train['Pre-Revision Incorrect Answer 3']


        qa_pairs = []
        cnt = 0
        for q, ca, ia1, ia2, ia3 in zip(
                questions,
                correct_answers,
                incorrect_answers_1,
                incorrect_answers_2,
                incorrect_answers_3,
        ):
            cnt += 1
            if self.max_samples and cnt > self.max_samples:
                break
            # 先打乱答案顺序
            answer_list = [
                (ca, True),
                (ia1, False),
                (ia2, False),
                (ia3, False),
            ]
            random.shuffle(answer_list)
            letters = ["A", "B", "C", "D"]
            options = []
            correct_letter = None
            for idx, (text, is_correct) in enumerate(answer_list):
                letter = letters[idx]
                options.append((letter, text, is_correct))
                if is_correct:
                    correct_letter = letter

            prompt = "You are answering a multiple-choice question.\n\n"
            prompt += f"Question:\n{q}\n\nChoices:\n"
            for letter, text, _ in options:
                prompt += f"{letter}. {text}\n"

            prompt += "\nRespond with only the letter (A, B, C, or D)."
            qa_pairs.append({"question": prompt, "answer": correct_letter})

        return qa_pairs
        import pdb; pdb.set_trace()
        '''
        train.column_names
        ['Pre-Revision Question', 'Pre-Revision Correct Answer', 'Pre-Revision Incorrect Answer 1', 'Pre-Revision Incorrect Answer 2', 'Pre-Revision Incorrect Answer 3', 
        'Pre-Revision Explanation', 'Self-reported question-writing time (minutes)', 'Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3', 
        'Explanation', 'Revision Comments (from Question Writer)', 'Subdomain', "Writer's Difficulty Estimate", 'Extra Revised Question', 'Extra Revised Explanation', 
        'Extra Revised Correct Answer', 'Extra Revised Incorrect Answer 1', 'Extra Revised Incorrect Answer 2', 'Extra Revised Incorrect Answer 3', 'Non-Expert Validator Accuracy', 
        'Majority Non-Expert Vals Incorrect', 'Expert Validator Accuracy', 'Record ID', 'High-level domain', 'Question Writer', 'Feedback_EV_1', 'Validator Revision Suggestion_EV_1', 
        'Is First Validation_EV_1', 'Post hoc agreement_EV_1', 'Sufficient Expertise?_EV_1', 'Understand the question?_EV_1', 'Question Difficulty_EV_1', 
        'Validator Answered Correctly_EV_1', 'Self-reported time (minutes)_EV_1', 'Probability Correct_EV_1', 'Manual Correctness Adjustment_EV_1', 'Expert Validator_EV_1', 
        'Feedback_EV_2', 'Validator Revision Suggestion_EV_2', 'Is First Validation_EV_2', 'Post hoc agreement_EV_2', 'Sufficient Expertise?_EV_2', 'Understand the question?_EV_2', 
        'Question Difficulty_EV_2', 'Validator Answered Correctly_EV_2', 'Self-reported time (minutes)_EV_2', 'Probability Correct_EV_2', 'Manual Correctness Adjustment_EV_2', 
        'Expert Validator_EV_2', 'Feedback_NEV_1', 'Validator Answered Correctly_NEV_1', 'Explanation_NEV_1', 'Self-reported time (minutes)_NEV_1', 'Websites visited_NEV_1', 
        'Probability Correct_NEV_1', 'Manual Correctness Adjustment_NEV_1', 'Non-Expert Validator_NEV_1', 'Feedback_NEV_2', 'Validator Answered Correctly_NEV_2', 'Explanation_NEV_2', 
        'Self-reported time (minutes)_NEV_2', 'Websites visited_NEV_2', 'Probability Correct_NEV_2', 'Manual Correctness Adjustment_NEV_2', 'Non-Expert Validator_NEV_2', 'Feedback_NEV_3', 
        'Validator Answered Correctly_NEV_3', 'Explanation_NEV_3', 'Self-reported time (minutes)_NEV_3', 'Websites visited_NEV_3', 'Probability Correct_NEV_3', 
        'Manual Correctness Adjustment_NEV_3', 'Non-Expert Validator_NEV_3', 'Expert Validator Disagreement Category', 'Canary String']
        '''
    

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
    loader = GPQALoader(base_path="", max_samples=1000)
    data = loader.load_data(split='train')
    