"""Chain-of-Thought reasoning strategy with step-by-step thinking."""

from typing import Dict, Any, List
import re
from .base import BaseStrategy, StrategyOutput


class ChainOfThoughtStrategy(BaseStrategy):
    """Chain-of-Thought reasoning with inline step-by-step planning."""
    
    def generate_prompt(self, question: str) -> str:
        """Generate CoT prompt encouraging step-by-step reasoning.
        
        Args:
            question: Input question/problem
            
        Returns:
            Formatted prompt for step-by-step reasoning
        """
        prompt_template = self.config.get('prompt_template',
            "Question: {question}\n\n"
            "Let's think step by step. And execute it at the same time.\n"
            "Format:"
            "Step 1. ..."
            "Step 2. ..."
            "..."
            "Step n. ... <END>"
        )
        return prompt_template.format(question=question)
    
    def execute(self, question: str, hook_layers, **kwargs) -> StrategyOutput:
        """Execute Chain-of-Thought reasoning strategy.
        
        Args:
            question: Input question/problem
            **kwargs: Additional arguments
            
        Returns:
            StrategyOutput with response and captured states
        """
        prompt = self.generate_prompt(question)
        

        response, hidden_states = self.generate_response_hidden(
            prompt=prompt,
            hook_layers=hook_layers,
            max_new_tokens=self.config.get('max_new_tokens'),
            **kwargs
        )

        # Parse steps from response
        steps = self._parse_steps(response)
        
        return StrategyOutput(
            response=response,
            hidden_states=hidden_states,
            steps=steps,
            metadata={
                'strategy': 'cot',
                'prompt': prompt,
                'answer': response
            }
        )
    
    def steer(self, question, hook_layers_idx, saes, alpha, steer_n_steps=1, **kwargs):
        prompt = self.generate_prompt(question)
        
        # Get prompt hidden states (what we actually want for analysis)
        # prompt_hidden_states = self.get_prompt_hidden_states(prompt)
        
        # Generate response (we still need the response but not its hidden states)
        # 
        response = self.generate_steered_response(
            prompt=prompt,
            hook_layers_idx=hook_layers_idx,
            max_new_tokens=self.config.get('max_new_tokens'),
            saes=saes,
            alpha=alpha,
            n_steps = steer_n_steps,
            **kwargs
        )

        # Parse response - remove <END> and everything after it
        # steps = self._parse_steps(response)
        
        return StrategyOutput(
            response=response,
            metadata={
                'strategy': 'direct',
                'prompt': prompt,
                'answer': response
            }
        )

    
    def _parse_steps(self, response: str) -> List[str]:
        """Parse reasoning steps from CoT response.
        
        Args:
            response: Generated response text
            
        Returns:
            List of reasoning steps
        """
        # Look for numbered steps or reasoning patterns
        step_patterns = [
            r'Step \d+:([^\\n]+)',
            r'\d+\.([^\\n]+)',
            r'First,([^\\n]+)',
            r'Then,([^\\n]+)',
            r'Next,([^\\n]+)',
            r'Finally,([^\\n]+)'
        ]
        
        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            steps.extend([match.strip() for match in matches])
        
        # If no explicit steps found, split by sentences
        if not steps:
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            steps = sentences[:5]  # Limit to reasonable number
        
        return steps