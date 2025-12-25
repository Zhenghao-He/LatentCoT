
from pathlib import Path
from typing import List, Dict, Any, Optional
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



    