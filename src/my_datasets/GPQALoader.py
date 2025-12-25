from .dataset_loader import DataLoader
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import random
from datasets import load_dataset
import re
from typing import Optional

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