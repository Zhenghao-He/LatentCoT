"""Simple dataset loader for experiments."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging




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

    def check_answer_correctness(self, predicted: str, ground_truth: str) -> bool:
        if not predicted or not ground_truth:
            return False
        pred_clean = predicted.strip().lower()
        truth_clean = ground_truth.strip().lower()
        if pred_clean == truth_clean:
            return True
        import re
        pred_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', pred_clean)
        truth_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', truth_clean)
        if pred_numbers and truth_numbers:
            return pred_numbers[-1] == truth_numbers[-1]
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