"""Utilities for extracting and evaluating answers from model responses."""

import re
from typing import Optional


def extract_answer(response: str) -> str:
    """Extract final answer from model response.
    
    Args:
        response: Generated response text
        
    Returns:
        Extracted answer string
    """
    if not response:
        return ""
    
    # Clean up the response - stop at <END> if present
    if '<END>' in response:
        response = response.split('<END>')[0]
    
    # Try to find the last meaningful line
    lines = response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('//'):
            return line
    
    return response.strip()


def check_answer_correctness(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth.
    
    Args:
        predicted: Model's predicted answer
        ground_truth: Correct answer
        
    Returns:
        True if answers match, False otherwise
    """
    if not predicted or not ground_truth:
        return False
    
    # Simple exact match
    pred_clean = predicted.strip().lower()
    truth_clean = ground_truth.strip().lower()
    
    # Try exact match first
    if pred_clean == truth_clean:
        return True
    
    # Try extracting numbers and comparing
    pred_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', pred_clean)
    truth_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', truth_clean)
    
    if pred_numbers and truth_numbers:
        return pred_numbers[-1] == truth_numbers[-1]  # Compare last number found
    
    return False