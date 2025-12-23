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
        num_generated_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.response = response
        self.hidden_states = hidden_states or []
        self.logits = logits
        self.num_generated_tokens = num_generated_tokens
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
        # if "<END>" not in self.tokenizer.get_vocab():
        #     special_tokens = {"additional_special_tokens": ["<END>"]}
        #     self.tokenizer.add_special_tokens(special_tokens)
        #     self.model.resize_token_embeddings(len(self.tokenizer))
    
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
    def steer(self, question: str, hook_layers_idx, k_index,saes, alpha, **kwargs) -> StrategyOutput:

        pass

    def generate_steered_response(
        self,
        prompt: str,
        hook_layers_idx: Dict[str, torch.Tensor],  # {layer_name: feature_indices}
        k_index: int,
        max_new_tokens: Optional[int] = None,
        saes: Dict[str, Any] = None,  # {layer_name: sae_model}
        alpha: float = 1.0,  # steering strength
        n_steps: int =1,
        **generation_kwargs
    ) -> str:
        """Generate response with SAE steering on specified layers.
        
        Args:
            prompt: Input prompt
            hook_layers_idx: Dict mapping layer names to feature indices to steer
            max_new_tokens: Maximum number of new tokens to generate
            saes: Dict mapping layer names to SAE models
            alpha: Steering strength multiplier
            **generation_kwargs: Additional generation arguments
            
        Returns:
            Generated text string
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.model.device)
        
        # get <END> token id
        # end_token_id = self.tokenizer.convert_tokens_to_ids("<END>")
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Set default generation parameters
        gen_kwargs = {
            'do_sample': False,
            'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            # 'stopping_criteria': self._get_stopping_criteria(end_token_id, len(inputs.input_ids[0])),
            'eos_token_id': terminators,
            'output_hidden_states': True,
            'return_dict_in_generate': True,
            'max_new_tokens': max_new_tokens,
            **generation_kwargs
        }
        
        handles = []
        
        # Register forward hooks for steering
        for layer_name, feature in hook_layers_idx.items():
            sae = saes.get(layer_name)
            if sae is None:
                print(f"Warning: No SAE found for layer {layer_name}, skipping")
                continue
            if hasattr(self.model, "hf_device_map"):
                # 找到该层对应的设备，例如 "cuda:1"
                target_device = self.model.hf_device_map.get(layer_name, self.device)
            else:
                target_device = self.device
            feature_acts, feature_idx = feature  
            feature_acts = torch.tensor(feature_acts, device=self.device, dtype=torch.float16)
            # feature_idx = torch.tensor(feature_idx, device=self.device, dtype=torch.float16)
            # Get the module to hook
            # import pdb; pdb.set_trace()
            module = self.model.base_model.get_submodule(layer_name)
            
            def make_steering_hook(sae_model, feature_acts, feature_idx, strength, n_steps=1):
                remaining = n_steps
                def hook_fn(module, inputs, outputs):
                    nonlocal remaining
                    if remaining <= 0:
                        return outputs
                    
                    hidden = outputs[0]  # [seq, hidden_dim] 取第一个batch
                    # import pdb; pdb.set_trace()
                    # Only steer the last token position
                    last_hidden = hidden[-1:, :]  # [1, hidden_dim]
                    
                    # Move SAE to same device and dtype
                    sae = sae_model.to(last_hidden.device)
                    sae_dtype = next(sae.parameters()).dtype
                    last_hidden_typed = last_hidden.to(sae_dtype)
                    # import pdb; pdb.set_trace()
                    # Encode to latent space
                    latent_tuple = sae.encode(last_hidden_typed)  # [batch, latent_dim]
                    if sae.__class__.__name__ == "SparseAutoEncoder":
                        latent_z = latent_tuple[0]
                        pre_acts = latent_tuple[1]
                        steered_pre_acts = pre_acts.clone()
                        # import pdb; pdb.set_trace()
                        # steered_pre_acts[:, feature_idx] += strength * feature_acts
                        steered_pre_acts[:, feature_idx] += strength
                        # steered_pre_acts[:, feature_idx] =0
                        steered_pre_acts = torch.nn.functional.relu(steered_pre_acts)

                        base_line = sae.decode(latent_z)
                        hidden_steered = sae.decode(steered_pre_acts)
                    else:

                        top_acts = latent_tuple[0]      # Top-k activation values
                        top_indices = latent_tuple[1]   # Top-k feature indices
                        pre_acts = latent_tuple[2]  # Pre-activation values (if applicable) 
                        steered_pre_acts = pre_acts.clone()
                        steered_pre_acts[:, feature_idx] += strength 
                        top_acts_steered, top_indices_steered = torch.topk(steered_pre_acts, k=top_acts.size(1), sorted=False)
                    
                    # Decode back to hidden space
                        hidden_steered = sae.decode(top_acts_steered, top_indices_steered)  # [batch, hidden_dim]
                        base_line = sae.decode(top_acts, top_indices)
                    increment = hidden_steered - base_line
                    hidden_steered = last_hidden_typed + increment
                    # import pdb; pdb.set_trace()
                    # Replace the last token's hidden state
                    hidden[-1, :] = hidden_steered.to(hidden.dtype)
                    outputs = hidden.unsqueeze(dim=0)  # Return as tuple
                    # Return modified output tuple
                    remaining -= 1
                    return outputs
                
                return hook_fn
            if k_index is None:
                h = module.register_forward_hook(make_steering_hook(sae, feature_acts, feature_idx[3:5], alpha, n_steps=n_steps))
            else:
                h = module.register_forward_hook(make_steering_hook(sae, feature_acts, [feature_idx[k_index]], alpha, n_steps=n_steps))
            handles.append(h)
        
        # Generate with steering
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **gen_kwargs
            )
        
        # Remove all hooks
        for h in handles:
            h.remove()
        
        # Decode generated text
        generated_ids = outputs.sequences[0][len(inputs.input_ids[0]):]
        num_generated_tokens = len(generated_ids)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        # Cleanup
        del outputs
        del inputs
        del generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return generated_text, num_generated_tokens

    def generate_response_hidden(
        self,
        prompt: str,
        hook_layers: List[str], 
        max_new_tokens: Optional[int] = None,
        **generation_kwargs
    ) -> Tuple[str, List[torch.Tensor]]:
        
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.device)
        
        # get <END> 的 id (已在__init__中添加)
        # end_token_id = self.tokenizer.convert_tokens_to_ids("<END>")

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        # Set default generation parameters
        gen_kwargs = {
            'do_sample': False,
            'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            # 'stopping_criteria': self._get_stopping_criteria(end_token_id, len(inputs.input_ids[0])),
            'eos_token_id': terminators,
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
        num_generated_tokens = len(generated_ids)
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

        return generated_text, hidden_states, num_generated_tokens
       


    # def _get_stopping_criteria(self, end_token_id: int, input_length: int) -> StoppingCriteriaList:
    #     """Create stopping criteria for <END> token.
        
    #     Args:
    #         end_token_id: Token ID for <END>
    #         input_length: Length of input prompt to exclude from checking
            
    #     Returns:
    #         StoppingCriteriaList containing custom stopping criteria
    #     """
    #     class EndTokenStoppingCriteria(StoppingCriteria):
    #         def __init__(self, end_token_id: int, tokenizer, input_length: int):
    #             self.end_token_id = end_token_id
    #             self.tokenizer = tokenizer
    #             self.input_length = input_length
            
    #         def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    #             del scores, kwargs  # Unused parameters
    #             # Only check the newly generated tokens, not the input prompt
    #             if input_ids.size(1) <= self.input_length:
    #                 return False
                    
    #             generated_tokens = input_ids[0, self.input_length:].tolist()
    #             if len(generated_tokens) == 0:
    #                 return False
                    
    #             decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
    #             # Only stop if we see a complete <END> token, not just partial matches
    #             contains_end = "<END>" in decoded_text and decoded_text.strip().endswith(">")
    #             # if contains_end:
    #             #     print(f"Found complete <END> in generated part: {repr(decoded_text)}")
    #             return contains_end
        
    #     return StoppingCriteriaList([EndTokenStoppingCriteria(end_token_id, self.tokenizer, input_length)])
    

    
    @property
    def name(self) -> str:
        """Get strategy name."""
        return self.__class__.__name__.replace('Strategy', '').lower()