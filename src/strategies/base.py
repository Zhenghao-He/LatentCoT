"""Base class for reasoning strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, StoppingCriteria, StoppingCriteriaList


class StrategyOutput:
    """Output container for strategy execution."""
    
    def __init__(
        self,
        response: str,
        hidden_states: Optional[List[torch.Tensor]] = None,
        logits: Optional[torch.Tensor] = None,
        steps: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.response = response
        self.hidden_states = hidden_states or []
        self.logits = logits
        self.steps = steps or []
        self.metadata = metadata or {}


class BaseStrategy(ABC):
    """Abstract base class for reasoning strategies."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any]
    ):
        """Initialize strategy with model and configuration.
        
        Args:
            model: Pre-trained language model
            tokenizer: Tokenizer for the model
            config: Strategy-specific configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        
        # Enable output of hidden states
        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = False
        
        # Add special tokens ONCE during initialization to prevent memory leaks
        if "<END>" not in self.tokenizer.get_vocab():
            special_tokens = {"additional_special_tokens": ["<END>"]}
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    @abstractmethod
    def generate_prompt(self, question: str) -> str:
        """Generate prompt for the given question.
        
        Args:
            question: Input question/problem
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def execute(self, question: str, **kwargs) -> StrategyOutput:
        """Execute the reasoning strategy on a question.
        
        Args:
            question: Input question/problem
            **kwargs: Additional arguments
            
        Returns:
            StrategyOutput containing response and analysis data
        """
        pass
    
    
    @abstractmethod
    def steer(self, question: str, hook_layers_idx, **kwargs):

        pass


    def generate_response_hidden(
        self,
        prompt: str,
        hook_layers: List[str], 
        max_new_tokens: Optional[int] = None,
        **generation_kwargs
    ) -> Tuple[str, List[torch.Tensor]]:
        
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.device)
        
        # get <END> 的 id (已在__init__中添加)
        end_token_id = self.tokenizer.convert_tokens_to_ids("<END>")


        # Set default generation parameters
        gen_kwargs = {
            'do_sample': False,
            'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            'stopping_criteria': self._get_stopping_criteria(end_token_id, len(inputs.input_ids[0])),
            'output_hidden_states': True,
            'return_dict_in_generate': True,
            'max_new_tokens': max_new_tokens,
            **generation_kwargs
        }
        
        # 3. 准备存每一层、每一步的激活： {layer_name: [step0_vec, step1_vec, ...]}
        activations = {}
        handles = []

        # 4. 注册 forward hook 到指定层
        for layer_spec in hook_layers:
            hook_name = layer_spec
            activations[hook_name] = []

            module = self.model.base_model.get_submodule(hook_name)

            def make_hook(name):
                def hook_fn(module, inputs, outputs):
                    last_hidden = outputs[0, -1, :].detach().clone()   # [hidden_dim]
                    activations[name].append(last_hidden)
                return hook_fn

            h = module.register_forward_hook(make_hook(hook_name))
            handles.append(h)

        # 5. 生成（hook 会在每一步 forward 时自动被调用）
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **gen_kwargs
            )

        # 6. 取消所有 hook
        for h in handles:
            h.remove()


        # 7. 解码生成文本
        generated_ids = outputs.sequences[0][len(inputs.input_ids[0]):]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        # 8. 把每层的 [step_i hidden] 列表堆成 tensor: [num_steps, hidden_dim]
        hidden_states = []
        for hook_name in hook_layers:
            layer_acts = activations[hook_name]
            if len(layer_acts) == 0:
                # 理论上不应该，但防一下空的
                hidden_states.append(torch.empty(0, self.model.config.hidden_size))
            else:
                hidden_states.append(torch.stack(layer_acts, dim=0))  # [num_steps, hidden]
        # import pdb; pdb.set_trace()
        # 9. 清理显存
        del outputs
        del inputs
        del generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return generated_text, hidden_states
       


    def _get_stopping_criteria(self, end_token_id: int, input_length: int) -> StoppingCriteriaList:
        """Create stopping criteria for <END> token.
        
        Args:
            end_token_id: Token ID for <END>
            input_length: Length of input prompt to exclude from checking
            
        Returns:
            StoppingCriteriaList containing custom stopping criteria
        """
        class EndTokenStoppingCriteria(StoppingCriteria):
            def __init__(self, end_token_id: int, tokenizer, input_length: int):
                self.end_token_id = end_token_id
                self.tokenizer = tokenizer
                self.input_length = input_length
            
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                del scores, kwargs  # Unused parameters
                # Only check the newly generated tokens, not the input prompt
                if input_ids.size(1) <= self.input_length:
                    return False
                    
                generated_tokens = input_ids[0, self.input_length:].tolist()
                if len(generated_tokens) == 0:
                    return False
                    
                decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                # Only stop if we see a complete <END> token, not just partial matches
                contains_end = "<END>" in decoded_text and decoded_text.strip().endswith(">")
                # if contains_end:
                #     print(f"Found complete <END> in generated part: {repr(decoded_text)}")
                return contains_end
        
        return StoppingCriteriaList([EndTokenStoppingCriteria(end_token_id, self.tokenizer, input_length)])
    

    
    @property
    def name(self) -> str:
        """Get strategy name."""
        return self.__class__.__name__.replace('Strategy', '').lower()