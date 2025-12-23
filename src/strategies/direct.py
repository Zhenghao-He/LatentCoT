"""Direct reasoning strategy without intermediate planning."""

from typing import Dict, Any
from .base import BaseStrategy, StrategyOutput


class DirectStrategy(BaseStrategy):
    """Direct answering without explicit planning or step-by-step reasoning."""
    
    def generate_prompt(self, question: str) -> str:
        """Generate direct prompt for immediate answering.
        
        Args:
            question: Input question/problem
            
        Returns:
            Formatted prompt for direct answering
        """
        # prompt_template = self.config.get('strategy.direct.prompt_template', 
        #     "Question: {question}\n\n"
        #     "Give me the answer directly without explanations.\n"
        #     "Format your response as: your answer here.<|eot_id|>"
        # )
        prompt_template = self.config.get('strategy.direct.prompt_template', 
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|>" # 最好加上 System Prompt 明确身份
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Question: {question}\n\n"
            "Give me the answer directly without explanations.<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n" # 注意这里加两个换行
        )
        return prompt_template.format(question=question)
    
    def steer(self, question, hook_layers_idx, k_index, saes, alpha, steer_n_steps=1, **kwargs):
        prompt = self.generate_prompt(question)
        
        # Get prompt hidden states (what we actually want for analysis)
        # prompt_hidden_states = self.get_prompt_hidden_states(prompt)
        
        # Generate response (we still need the response but not its hidden states)
        # 
        response, num_generated_tokens = self.generate_steered_response(
            prompt=prompt,
            hook_layers_idx=hook_layers_idx,
            k_index=k_index,
            max_new_tokens=self.config.get('max_new_tokens'),
            saes=saes,
            alpha=alpha,
            n_steps = steer_n_steps,
            **kwargs
        )

        # Parse response - remove <END> and everything after it
        parsed_response = self._parse_response(response)
        
        return StrategyOutput(
            response=response,
            num_generated_tokens=num_generated_tokens,
            metadata={
                'strategy': 'direct',
                'prompt': prompt,
                'answer': parsed_response
            }
        )

    def execute(self, question: str, hook_layers, **kwargs) -> StrategyOutput:
        """Execute direct reasoning strategy.
        
        Args:
            question: Input question/problem
            hook_layers: Layers to hook for capturing hidden states
            
        Returns:
            StrategyOutput with response and captured states
        """
        # import pdb; pdb.set_trace()
        prompt = self.generate_prompt(question)
        
        # Get prompt hidden states (what we actually want for analysis)
        # prompt_hidden_states = self.get_prompt_hidden_states(prompt)
        
        # Generate response (we still need the response but not its hidden states)
        # 
        response, hidden_states, num_generated_tokens = self.generate_response_hidden(
            prompt=prompt,
            hook_layers=hook_layers,
            max_new_tokens=self.config.get('max_new_tokens'),
            **kwargs
        )

        # Parse response - remove <|eot_id|> and everything after it
        parsed_response = self._parse_response(response)
        
        return StrategyOutput(
            response=response,
            hidden_states=hidden_states,
            num_generated_tokens=num_generated_tokens,
            metadata={
                'strategy': 'direct',
                'prompt': prompt,
                'answer': parsed_response
            }
        )
    
    def _parse_response(self, response: str) -> str:
        """Parse response by removing <END> and everything after it.
        
        Args:
            response: Raw generated response
            
        Returns:
            Cleaned response without <|eot_id|> token
        """
        if '<|eot_id|>' in response:
            return response.split('<|eot_id|>')[0].strip()
        return response.strip()