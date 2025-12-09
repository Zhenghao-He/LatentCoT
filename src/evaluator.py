"""Evaluation utilities for strategy performance assessment."""

import re
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


class StrategyEvaluator:
    """Evaluator for strategy performance and correctness."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate_responses(
        self, 
        questions: List[str], 
        responses: List[str],
        ground_truth: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate strategy responses.
        
        Args:
            questions: List of input questions
            responses: List of generated responses
            ground_truth: Optional ground truth answers
            
        Returns:
            Evaluation metrics
        """
        if len(questions) != len(responses):
            logger.warning("Mismatch between questions and responses length")
            min_len = min(len(questions), len(responses))
            questions = questions[:min_len]
            responses = responses[:min_len]
        
        results = {
            'num_questions': len(questions),
            'response_lengths': self._analyze_response_lengths(responses),
            'answer_extraction': self._analyze_answer_extraction(responses),
            'reasoning_quality': self._analyze_reasoning_quality(responses)
        }
        
        # Add ground truth comparison if available
        if ground_truth:
            if len(ground_truth) != len(responses):
                logger.warning("Ground truth length mismatch")
                min_len = min(len(ground_truth), len(responses))
                ground_truth = ground_truth[:min_len]
                responses = responses[:min_len]
            
            results['accuracy'] = self._compute_accuracy(responses, ground_truth)
        
        return results
    
    def _analyze_response_lengths(self, responses: List[str]) -> Dict[str, float]:
        """Analyze response length statistics.
        
        Args:
            responses: List of response strings
            
        Returns:
            Length statistics
        """
        lengths = [len(response.split()) for response in responses]
        
        return {
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'median_length': np.median(lengths)
        }
    
    def _analyze_answer_extraction(self, responses: List[str]) -> Dict[str, Any]:
        """Analyze how well answers can be extracted from responses.
        
        Args:
            responses: List of response strings
            
        Returns:
            Answer extraction analysis
        """
        extraction_results = {
            'extractable_answers': 0,
            'numeric_answers': 0,
            'answer_patterns': Counter(),
            'extraction_confidence': []
        }
        
        for response in responses:
            # Try to extract numerical answers
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
            
            # Look for answer indicators
            answer_patterns = [
                r'[Aa]nswer:?\s*(.+)',
                r'[Tt]he answer is\s*(.+)',
                r'[Ss]o\s*(.+)',
                r'[Tt]herefore\s*(.+)'
            ]
            
            extracted = False
            confidence = 0.0
            
            for pattern in answer_patterns:
                matches = re.findall(pattern, response)
                if matches:
                    extraction_results['answer_patterns'][pattern] += 1
                    extracted = True
                    confidence += 0.5
            
            if numbers:
                extraction_results['numeric_answers'] += 1
                extracted = True
                confidence += 0.3
            
            if extracted:
                extraction_results['extractable_answers'] += 1
            
            extraction_results['extraction_confidence'].append(confidence)
        
        # Compute average confidence
        avg_confidence = (
            np.mean(extraction_results['extraction_confidence'])
            if extraction_results['extraction_confidence'] else 0.0
        )
        
        return {
            'extractable_rate': extraction_results['extractable_answers'] / len(responses),
            'numeric_rate': extraction_results['numeric_answers'] / len(responses),
            'avg_extraction_confidence': avg_confidence,
            'common_patterns': dict(extraction_results['answer_patterns'].most_common(3))
        }
    
    def _analyze_reasoning_quality(self, responses: List[str]) -> Dict[str, Any]:
        """Analyze quality of reasoning in responses.
        
        Args:
            responses: List of response strings
            
        Returns:
            Reasoning quality metrics
        """
        quality_metrics = {
            'step_indicators': 0,
            'logical_connectors': 0,
            'calculation_steps': 0,
            'reasoning_depth': []
        }
        
        # Patterns for reasoning quality
        step_patterns = [r'[Ss]tep \d+', r'\d+\.', r'[Ff]irst', r'[Tt]hen', r'[Nn]ext', r'[Ff]inally']
        logical_patterns = [r'[Bb]ecause', r'[Ss]ince', r'[Tt]herefore', r'[Ss]o', r'[Hh]ence']
        calculation_patterns = [r'\d+\s*[+\-*/]\s*\d+', r'=', r'\$\d+']
        
        for response in responses:
            # Count step indicators
            step_count = sum(len(re.findall(pattern, response)) for pattern in step_patterns)
            if step_count > 0:
                quality_metrics['step_indicators'] += 1
            
            # Count logical connectors
            logical_count = sum(len(re.findall(pattern, response)) for pattern in logical_patterns)
            if logical_count > 0:
                quality_metrics['logical_connectors'] += 1
            
            # Count calculation steps
            calc_count = sum(len(re.findall(pattern, response)) for pattern in calculation_patterns)
            if calc_count > 0:
                quality_metrics['calculation_steps'] += 1
            
            # Estimate reasoning depth (number of sentences)
            sentences = len([s for s in response.split('.') if s.strip()])
            quality_metrics['reasoning_depth'].append(sentences)
        
        return {
            'step_indicator_rate': quality_metrics['step_indicators'] / len(responses),
            'logical_connector_rate': quality_metrics['logical_connectors'] / len(responses),
            'calculation_step_rate': quality_metrics['calculation_steps'] / len(responses),
            'avg_reasoning_depth': np.mean(quality_metrics['reasoning_depth']),
            'std_reasoning_depth': np.std(quality_metrics['reasoning_depth'])
        }
    
    def _compute_accuracy(self, responses: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """Compute accuracy against ground truth.
        
        Args:
            responses: Generated responses
            ground_truth: Ground truth answers
            
        Returns:
            Accuracy metrics
        """
        exact_matches = 0
        numeric_matches = 0
        
        for response, truth in zip(responses, ground_truth):
            # Exact match (after normalization)
            if self._normalize_answer(response) == self._normalize_answer(truth):
                exact_matches += 1
            
            # Numeric match (extract and compare numbers)
            response_nums = self._extract_numbers(response)
            truth_nums = self._extract_numbers(truth)
            
            if response_nums and truth_nums:
                # Check if any extracted number matches
                if any(abs(r - t) < 1e-6 for r in response_nums for t in truth_nums):
                    numeric_matches += 1
        
        return {
            'exact_accuracy': exact_matches / len(responses),
            'numeric_accuracy': numeric_matches / len(responses)
        }
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer string for comparison.
        
        Args:
            answer: Answer string
            
        Returns:
            Normalized answer
        """
        # Remove extra whitespace and convert to lowercase
        normalized = re.sub(r'\s+', ' ', answer.strip().lower())
        
        # Remove common punctuation
        normalized = re.sub(r'[.,!?;:]', '', normalized)
        
        return normalized
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted numbers
        """
        # Find all numbers (including decimals)
        number_matches = re.findall(r'-?\d+(?:\.\d+)?', text)
        
        try:
            numbers = [float(match) for match in number_matches]
            return numbers
        except ValueError:
            return []
    
    def compare_strategies(
        self, 
        strategy_evaluations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare evaluation results across strategies.
        
        Args:
            strategy_evaluations: Evaluations for each strategy
            
        Returns:
            Comparison results
        """
        comparison = {
            'rankings': {},
            'differences': {},
            'summary': {}
        }
        
        # Extract metrics for comparison
        metrics_to_compare = [
            'response_lengths.mean_length',
            'answer_extraction.extractable_rate',
            'reasoning_quality.avg_reasoning_depth'
        ]
        
        # Add accuracy metrics if available
        if any('accuracy' in eval_result for eval_result in strategy_evaluations.values()):
            metrics_to_compare.extend(['accuracy.exact_accuracy', 'accuracy.numeric_accuracy'])
        
        for metric_path in metrics_to_compare:
            metric_values = {}
            
            for strategy, evaluation in strategy_evaluations.items():
                # Navigate nested dictionary
                value = evaluation
                for key in metric_path.split('.'):
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        value = None
                        break
                
                if value is not None:
                    metric_values[strategy] = value
            
            if len(metric_values) > 1:
                # Rank strategies
                sorted_strategies = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                comparison['rankings'][metric_path] = sorted_strategies
                
                # Compute differences
                max_val = max(metric_values.values())
                min_val = min(metric_values.values())
                comparison['differences'][metric_path] = {
                    'absolute_diff': max_val - min_val,
                    'relative_diff': (max_val - min_val) / (max_val + 1e-8)
                }
                
                comparison['summary'][metric_path] = metric_values
        
        return comparison