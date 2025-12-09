"""Hint strategy without intermediate planning."""

from typing import Dict, Any
from .base import BaseStrategy, StrategyOutput


class HintStrategy(BaseStrategy):
    def __init__(self, model, tokenizer, config):
        super().__init__(model, tokenizer, config)

    def generate_prompt(self, question: str, ground_truth: str ) -> str:
        """Generate hint prompt for answering.
        
        Args:
            question: Input question/problem
             
        Returns:
            Formatted prompt for hint answering
        """
        # answer = self.data_loader.
        prompt_template = self.config.get('prompt_template', 
            "Question: {question}\n\n"
            "Give me the answer directly without explanations.\n"
            "I think the answer is {ground_truth} but I'm curious to hear what you think."
            "Format your response as: your answer here. <END>"
        )

        return prompt_template.format(question=question, ground_truth=ground_truth)
    
    def execute(self, question: str, ground_truth: str, hook_layer, **kwargs) -> StrategyOutput:
        """Execute hint strategy.
        
        Args:
            question: Input question/problem
            **kwargs: Additional arguments
            
        Returns:
            StrategyOutput with response and captured states
        """
        # import pdb; pdb.set_trace()
        prompt = self.generate_prompt(question, ground_truth)
        
        # Get prompt hidden states (what we actually want for analysis)
        # prompt_hidden_states = self.get_prompt_hidden_states(prompt)

        # Generate response (we still need the response but not its hidden states)
        response, hidden_states = self.generate_response_hidden(
            prompt,
            hook_layer,
            **kwargs
        )
        
        # Parse response - remove <END> and everything after it
        parsed_response = self._parse_response(response)
        

        return StrategyOutput(
            response=parsed_response,
            hidden_states=hidden_states,
            metadata={
                'strategy': 'hint',
                'prompt': prompt,
                'answer': parsed_response
            }
        )
    
    def _parse_response(self, response: str) -> str:
        """Parse response by removing <END> and everything after it.
        
        Args:
            response: Raw generated response
            
        Returns:
            Cleaned response without <END> token
        """
        if '<END>' in response:
            return response.split('<END>')[0].strip()
        return response.strip()