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

    def generate_to_get_activations(
        self,
        prompt: str,
        features: Dict[str, torch.Tensor],  # {layer_name: feature_indices}
        k_index: int,
        max_new_tokens: Optional[int] = None,
        saes: Dict[str, Any] = None,  # {layer_name: sae_model}
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
        # terminators = [
        #     self.tokenizer.eos_token_id,
        #     self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]
        stop_tokens = ["<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<end_of_turn>",  "<|end▁of▁sentence|>"]
        terminators = [self.tokenizer.eos_token_id]  # 默认包含通用 eos
        for token_text in stop_tokens:
            t_id = self.tokenizer.convert_tokens_to_ids(token_text)
            if t_id is not None:
                terminators.append(t_id)

        # 去重，防止重复 ID
        terminators = list(set(terminators))

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
        activations = {}

       
        # Register forward hooks for steering
        for layer_name, feature in features.items():
            sae = saes.get(layer_name)
            if sae is None:
                print(f"Warning: No SAE found for layer {layer_name}, skipping")
                continue

            if hasattr(self.model, "hf_device_map"):
                # 找到该层对应的设备，例如 "cuda:1"
                if hasattr(self.model.base_model, "language_model"):
                    # layer_module = self.model.base_model.language_model.get_submodule(layer_name)
                    target_device = self.model.hf_device_map.get("model.language_model." + layer_name, self.device)
                else:
                    target_device = self.model.hf_device_map.get("model." + layer_name, self.device)
            else:
                target_device = self.device
            _, feature_idx = feature  

            feature_idx = torch.tensor(feature_idx, device=target_device, dtype=torch.long)
            sae = sae.to(target_device)
            if hasattr(self.model.base_model, "language_model"):
                module = self.model.base_model.language_model.get_submodule(layer_name)
            else:
                module = self.model.base_model.get_submodule(layer_name)

            activations[layer_name] = []

            def make_act_hook(layer_name, sae_model,  feature_idx):
                def hook_fn(module, inputs, outputs):
                    # import pdb; pdb.set_trace()
                    tuple_flag = isinstance(outputs, tuple)
                    if tuple_flag:
                        hidden = outputs[0][0]
                    else:
                        hidden = outputs[0]  # [seq, hidden_dim] 取第一个batch

                    last_hidden = hidden[-1:, :]  # [1, hidden_dim]
                    
                    sae_dtype = next(sae_model.parameters()).dtype
                    last_hidden_typed = last_hidden.to(sae_dtype)
                    latent_tuple = sae_model.encode(last_hidden_typed)  # [batch, latent_dim]
                    if sae_model.__class__.__name__ == "SparseAutoEncoder" or sae_model.__class__.__name__ == "JumpReLUSAE":
                        latent_z = latent_tuple[0]
                        res = latent_z.detach().cpu()
                        res = res[:, feature_idx].squeeze(0)
                        activations[layer_name].append(res.detach().float().cpu())

                    else:
                        raise NotImplementedError
                        top_acts = latent_tuple[0]      # Top-k activation values
                        top_indices = latent_tuple[1]   # Top-k feature indices
                        pre_acts = latent_tuple[2]  # Pre-activation values (if applicable) 
                        steered_pre_acts = pre_acts.clone()
                        mean_feat = torch.mean(torch.abs(steered_pre_acts[:, feature_idx]))
                        top_acts_steered, top_indices_steered = torch.topk(steered_pre_acts, k=top_acts.size(1), sorted=False)
                    
                    return outputs
                return hook_fn

            if k_index is None:
                h = module.register_forward_hook(make_act_hook(layer_name, sae, feature_idx[:3]))
            else:
                h = module.register_forward_hook(make_act_hook(layer_name, sae, [feature_idx[k_index]]))
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
        
        # Cleanup
        del outputs
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return activations


    def generate_to_get_raw_activations(
        self,
        prompt: str,
        hook_layers: List[str],
        max_new_tokens: Optional[int] = None,
        **generation_kwargs
    ) -> Dict[str, List[torch.Tensor]]:
        """Get raw hidden activations for the prompt's last token.
        
        Args:
            prompt: Input prompt
            hook_layers: Layer names to hook
            max_new_tokens: Unused, kept for call-site compatibility
            **generation_kwargs: Additional forward arguments
            
        Returns:
            Dict[layer_name, [last_token_hidden]], where hidden has shape
            [1, hidden_dim].
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.model.device)

        forward_kwargs = {
            'output_hidden_states': True,
            'return_dict': True,
            'use_cache': False,
            **generation_kwargs
        }
        
        handles = []
        activations = {}

       
        # Register forward hooks and capture the prompt's last token.
        for layer_name in hook_layers:
    

            if hasattr(self.model.base_model, "language_model"):
                module = self.model.base_model.language_model.get_submodule(layer_name)
            else:
                module = self.model.base_model.get_submodule(layer_name)

            activations[layer_name] = []

            def make_act_hook(layer_name):
                def hook_fn(module, inputs, outputs):
                    if isinstance(outputs, tuple):
                        hidden = outputs[0]
                    else:
                        hidden = outputs
                    # import pdb; pdb.set_trace()
                    last_hidden = hidden[0, -1, :].detach().float().cpu()  # [1, hidden_dim]
                    activations[layer_name].append(last_hidden)
                    return outputs
                return hook_fn

            h = module.register_forward_hook(make_act_hook(layer_name))
            handles.append(h)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **forward_kwargs
            )
        
        # Remove all hooks
        for h in handles:
            h.remove()
        
        # Cleanup
        del outputs
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return activations


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
        # terminators = [
        #     self.tokenizer.eos_token_id,
        #     self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]
        stop_tokens = ["<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end▁of▁sentence|>"]
        terminators = [self.tokenizer.eos_token_id]  # 默认包含通用 eos
        for token_text in stop_tokens:
            t_id = self.tokenizer.convert_tokens_to_ids(token_text)
            if t_id is not None:
                terminators.append(t_id)

        # 去重，防止重复 ID
        terminators = list(set(terminators))

        # Set default generation parameters
        gen_kwargs = {
            'do_sample': False,
            'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            # 'stopping_criteria': self._get_stopping_criteria(end_token_id, len(inputs.input_ids[0])),
            'eos_token_id': terminators,
            'output_hidden_states': True,
            'output_scores': True,
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
            # import pdb; pdb.set_trace()
            if hasattr(self.model, "hf_device_map"):
                # 找到该层对应的设备，例如 "cuda:1"
                if hasattr(self.model.base_model, "language_model"):
                    # layer_module = self.model.base_model.language_model.get_submodule(layer_name)
                    target_device = self.model.hf_device_map.get("model.language_model." + layer_name, self.device)
                else:
                    target_device = self.model.hf_device_map.get("model." + layer_name, self.device)
            else:
                target_device = self.device
            feature_acts, feature_idx = feature  
            # import pdb; pdb.set_trace()
            # if 
            # feature_acts = torch.tensor(feature_acts, device=target_device, dtype=torch.float16)
            feature_idx = torch.tensor(feature_idx, device=target_device, dtype=torch.long)
            sae = sae.to(target_device)
            # feature_idx = torch.tensor(feature_idx, device=self.device, dtype=torch.float16)
            # Get the module to hook
            # import pdb; pdb.set_trace()
            # module = self.model.base_model.get_submodule(layer_name)
            if hasattr(self.model.base_model, "language_model"):
                module = self.model.base_model.language_model.get_submodule(layer_name)
            else:
                module = self.model.base_model.get_submodule(layer_name)

            
            def make_steering_hook(sae_model, feature_acts, feature_idx, strength, n_steps=1):
                remaining = n_steps if n_steps is not None and n_steps > 0 else None
                def hook_fn(module, inputs, outputs):
                    nonlocal remaining
                    # import pdb; pdb.set_trace()
                    if remaining is None:
                        pass
                    elif remaining <= 0:
                        return outputs
                    elif remaining == 1:
                        pass
                    else:
                        # print("remaining:", remaining)
                        remaining -= 1
                        return outputs
                    # import pdb; pdb.set_trace()
                    # print(f"Steering at layer {layer_name}, remaining steps: {remaining}")
                    tuple_flag = isinstance(outputs, tuple)
                    if tuple_flag:
                        outputs = outputs[0]
                    hidden = outputs[0]  # [seq, hidden_dim] 取第一个batch
                    # import pdb; pdb.set_trace()
                    # Only steer the last token position
                    if remaining is None:
                        last_hidden = hidden[-256:, :]  # [1, hidden_dim]
                    else:
                        last_hidden = hidden[-1:, :]  # [1, hidden_dim]
                    
                    # Move SAE to same device and dtype
                    # sae = sae_model.to(last_hidden.device)
                    sae_dtype = next(sae_model.parameters()).dtype
                    last_hidden_typed = last_hidden.to(sae_dtype)
                    # import pdb; pdb.set_trace()
                    # Encode to latent space
                    latent_tuple = sae_model.encode(last_hidden_typed)  # [batch, latent_dim]
                    if sae_model.__class__.__name__ == "SparseAutoEncoder":
                        latent_z = latent_tuple[0]
                        pre_acts = latent_tuple[1]
                        steered_pre_acts = pre_acts.clone()
                        # import pdb; pdb.set_trace()
                        # steered_pre_acts[:, feature_idx] += strength * feature_acts
                        
                        
                        if remaining is None:
                            steered_pre_acts[:, feature_idx] = -1
                        else:
                            mean_feat = torch.mean(torch.abs(steered_pre_acts[:, feature_idx]))
                            steered_pre_acts[:, feature_idx] += strength* mean_feat
                        # steered_pre_acts[:, feature_idx] =0
                        steered_pre_acts = torch.nn.functional.relu(steered_pre_acts)

                        base_line = sae_model.decode(latent_z)
                        hidden_steered = sae_model.decode(steered_pre_acts)
                    elif sae_model.__class__.__name__ == "JumpReLUSAE":
                        latent_z = latent_tuple[0]
                        pre_acts = latent_tuple[1]
                        # mask = latent_tuple[2]
                        steered_pre_acts = pre_acts.clone()
                        # import pdb; pdb.set_trace()
                        # steered_pre_acts[:, feature_idx] += strength * feature_acts
                        if remaining is None:
                            steered_pre_acts[:, feature_idx] = -1000000
                        else:
                            mean_feat = torch.mean(torch.abs(steered_pre_acts[:, feature_idx]))
                            steered_pre_acts[:, feature_idx] += strength* mean_feat
                        # steered_pre_acts[:, feature_idx] = steered_pre_acts[:, feature_idx] + strength
                        mask = (steered_pre_acts > sae_model.threshold)
                        steered_pre_acts = mask * torch.nn.functional.relu(steered_pre_acts)
                        
                        # steered_pre_acts = torch.nn.functional.relu(steered_pre_acts)
                        base_line = sae_model.decode(latent_z)
                        hidden_steered = sae_model.decode(steered_pre_acts)
                        # import pdb; pdb.set_trace()
                    else:
                        # import pdb; pdb.set_trace()
                        top_acts = latent_tuple[0]      # Top-k activation values
                        top_indices = latent_tuple[1]   # Top-k feature indices
                        pre_acts = latent_tuple[2]  # Pre-activation values (if applicable) 
                        steered_pre_acts = pre_acts.clone()
                        if remaining is None:
                            steered_pre_acts[:, feature_idx] = -100000
                        else:
                            mean_feat = torch.mean(torch.abs(steered_pre_acts[:, feature_idx]))
                            steered_pre_acts[:, feature_idx] += strength* mean_feat
                        top_acts_steered, top_indices_steered = torch.topk(steered_pre_acts, k=top_acts.size(1), sorted=False)
                    
                    # Decode back to hidden space
                        hidden_steered = sae_model.decode(top_acts_steered, top_indices_steered)  # [batch, hidden_dim]
                        base_line = sae_model.decode(top_acts, top_indices)
                    increment = hidden_steered - base_line
                    hidden_steered = last_hidden_typed + increment
                    # import pdb; pdb.set_trace()
                    if remaining is None:
                        hidden_steered = hidden_steered[-1, :]  # [hidden_dim]
                    # Replace the last token's hidden state
                    hidden[-1, :] = hidden_steered.to(hidden.dtype)
                    outputs = hidden.unsqueeze(dim=0)  # Return as tuple
                    # Return modified output tuple
                    if remaining is not None:
                        remaining -= 1
                    # import pdb; pdb.set_trace()
                    if tuple_flag:
                        outputs = (outputs,)
                    return outputs
                
                return hook_fn
            if k_index is None:
                h = module.register_forward_hook(make_steering_hook(sae, feature_acts, feature_idx[:6], alpha, n_steps=n_steps))
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
        # import pdb; pdb.set_trace()
        # first_step_logits = outputs.scores[0][0]
        # first_probs  = torch.softmax(first_step_logits, dim=-1)
        # top_probs, top_ids = torch.topk(first_probs, k=10)
        # top_tokens = [self.tokenizer.decode([i]) for i in top_ids.tolist()]
        # second_step_logits = outputs.scores[1][0]
        # second_probs  = torch.softmax(second_step_logits, dim=-1)
        # top_probs_2, top_ids_2 = torch.topk(second_probs, k=10)
        # top_tokens_2 = [self.tokenizer.decode([i]) for i in top_ids_2.tolist()]
        # import pdb; pdb.set_trace()
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


    def generate_anti_steered_response(
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
        # terminators = [
        #     self.tokenizer.eos_token_id,
        #     self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]
        stop_tokens = ["<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<end_of_turn>",  "<|end▁of▁sentence|>"]
        terminators = [self.tokenizer.eos_token_id]  # 默认包含通用 eos
        for token_text in stop_tokens:
            t_id = self.tokenizer.convert_tokens_to_ids(token_text)
            if t_id is not None:
                terminators.append(t_id)

        # 去重，防止重复 ID
        terminators = list(set(terminators))

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
            # import pdb; pdb.set_trace()
            if hasattr(self.model, "hf_device_map"):
                # 找到该层对应的设备，例如 "cuda:1"
                if hasattr(self.model.base_model, "language_model"):
                    # layer_module = self.model.base_model.language_model.get_submodule(layer_name)
                    target_device = self.model.hf_device_map.get("model.language_model." + layer_name, self.device)
                else:
                    target_device = self.model.hf_device_map.get("model." + layer_name, self.device)
            else:
                target_device = self.device
            feature_acts, feature_idx = feature  
            # import pdb; pdb.set_trace()
            # if 
            # feature_acts = torch.tensor(feature_acts, device=target_device, dtype=torch.float16)
            feature_idx = torch.tensor(feature_idx, device=target_device, dtype=torch.long)
            sae = sae.to(target_device)
            # feature_idx = torch.tensor(feature_idx, device=self.device, dtype=torch.float16)
            # Get the module to hook
            # import pdb; pdb.set_trace()
            # module = self.model.base_model.get_submodule(layer_name)
            if hasattr(self.model.base_model, "language_model"):
                module = self.model.base_model.language_model.get_submodule(layer_name)
            else:
                module = self.model.base_model.get_submodule(layer_name)

            
            def make_steering_hook(sae_model, feature_acts, feature_idx, strength, n_steps=1):
                remaining = n_steps if n_steps is not None and n_steps > 0 else None
                def hook_fn(module, inputs, outputs):
                    nonlocal remaining
                    # import pdb; pdb.set_trace()
                    if remaining is None:
                        pass
                    elif remaining <= 0:
                        return outputs
                    elif remaining == 1:
                        pass
                    else:
                        # print("remaining:", remaining)
                        remaining -= 1
                        return outputs
                    # import pdb; pdb.set_trace()
                    # print(f"Steering at layer {layer_name}, remaining steps: {remaining}")
                    tuple_flag = isinstance(outputs, tuple)
                    if tuple_flag:
                        outputs = outputs[0]
                    hidden = outputs[0]  # [seq, hidden_dim] 取第一个batch
                    # import pdb; pdb.set_trace()
                    # Only steer the last token position
                    if remaining is None:
                        last_hidden = hidden[-256:, :]  # [1, hidden_dim]
                    else:
                        last_hidden = hidden[-1:, :]  # [1, hidden_dim]
                    
                    # Move SAE to same device and dtype
                    # sae = sae_model.to(last_hidden.device)
                    sae_dtype = next(sae_model.parameters()).dtype
                    last_hidden_typed = last_hidden.to(sae_dtype)
                    # import pdb; pdb.set_trace()
                    # Encode to latent space
                    
                    if sae_model.__class__.__name__ == "SparseAutoEncoder":
                        latent_tuple = sae_model.encode(last_hidden_typed)  # [batch, latent_dim]
                        latent_z = latent_tuple[0]
                        pre_acts = latent_tuple[1]
                        steered_pre_acts = pre_acts.clone()
                        import pdb; pdb.set_trace()
                        # steered_pre_acts[:, feature_idx] += strength * feature_acts
                        
                        
                        if remaining is None:
                            steered_pre_acts[:, feature_idx] = -1
                        else:
                            mean_feat = torch.mean(torch.abs(steered_pre_acts[:, feature_idx]))
                            steered_pre_acts[:, feature_idx] += strength* mean_feat
                        # steered_pre_acts[:, feature_idx] =0
                        steered_pre_acts = torch.nn.functional.relu(steered_pre_acts)

                        base_line = sae_model.decode(latent_z)
                        hidden_steered = sae_model.decode(steered_pre_acts)
                    elif sae_model.__class__.__name__ == "JumpReLUSAE":
                        latent_tuple = sae_model.encode(last_hidden_typed)  # [batch, latent_dim]
                        latent_z = latent_tuple[0]
                        pre_acts = latent_tuple[1]
                        # mask = latent_tuple[2]
                        steered_pre_acts = pre_acts.clone()
                        # import pdb; pdb.set_trace()
                        # steered_pre_acts[:, feature_idx] += strength * feature_acts
                        if remaining is None:
                            steered_pre_acts[:, feature_idx] = -1000000
                        else:
                            mean_feat = torch.mean(torch.abs(steered_pre_acts[:, feature_idx]))
                            steered_pre_acts[:, feature_idx] += strength* mean_feat
                        # steered_pre_acts[:, feature_idx] = steered_pre_acts[:, feature_idx] + strength
                        mask = (steered_pre_acts > sae_model.threshold)
                        steered_pre_acts = mask * torch.nn.functional.relu(steered_pre_acts)
                        latent_z = latent_tuple[0]
                        pre_acts = latent_tuple[1]
                        # steered_pre_acts = torch.nn.functional.relu(steered_pre_acts)
                        base_line = sae_model.decode(latent_z)
                        hidden_steered = sae_model.decode(steered_pre_acts)
                        # import pdb; pdb.set_trace()
                    elif sae_model.__class__.__name__ == "StandardSAE":
                        out, cache = sae_model.run_with_cache(last_hidden_typed)
                        latent_z = cache['hook_sae_acts_post']
                        pre_acts = cache['hook_sae_acts_pre']
                        steered_pre_acts = pre_acts.clone()
                        if remaining is None:
                            steered_pre_acts[:, feature_idx] = -1000000
                        else:
                            mean_feat = torch.mean(torch.abs(steered_pre_acts[:, feature_idx]))
                            steered_pre_acts[:, feature_idx] += strength* mean_feat

                        steered_pre_acts = torch.nn.functional.relu(steered_pre_acts)

                        base_line = out
                        hidden_steered = sae_model.decode(steered_pre_acts)
                    else:
                        # import pdb; pdb.set_trace()
                        latent_tuple = sae_model.encode(last_hidden_typed)  # [batch, latent_dim]
                        top_acts = latent_tuple[0]      # Top-k activation values
                        top_indices = latent_tuple[1]   # Top-k feature indices
                        pre_acts = latent_tuple[2]  # Pre-activation values (if applicable) 
                        steered_pre_acts = pre_acts.clone()
                        if remaining is None:
                            # import pdb; pdb.set_trace()
                            steered_pre_acts[:, feature_idx] = 0
                        else:
                            mean_feat = torch.mean(torch.abs(steered_pre_acts[:, feature_idx]))
                            steered_pre_acts[:, feature_idx] += strength* mean_feat
                        top_acts_steered, top_indices_steered = torch.topk(steered_pre_acts, k=top_acts.size(1), sorted=False)
                    
                    # Decode back to hidden space
                        hidden_steered = sae_model.decode(top_acts_steered, top_indices_steered)  # [batch, hidden_dim]
                        base_line = sae_model.decode(top_acts, top_indices)
                    increment = hidden_steered - base_line
                    hidden_steered = last_hidden_typed + increment
                    # import pdb; pdb.set_trace()
                    if remaining is None:
                        hidden_steered = hidden_steered[-1, :]  # [hidden_dim]
                    # Replace the last token's hidden state
                    hidden[-1, :] = hidden_steered.to(hidden.dtype)
                    outputs = hidden.unsqueeze(dim=0)  # Return as tuple
                    # Return modified output tuple
                    if remaining is not None:
                        remaining -= 1
                    # import pdb; pdb.set_trace()
                    if tuple_flag:
                        outputs = (outputs,)
                    return outputs
                
                return hook_fn
            if k_index is None:
                h = module.register_forward_hook(make_steering_hook(sae, feature_acts, feature_idx[:1], alpha, n_steps=n_steps))
            else:
                h = module.register_forward_hook(make_steering_hook(sae, feature_acts, [feature_idx[k_index]], alpha, n_steps=n_steps))
            handles.append(h)
        
        # ---- Token-wise prefill: feed prompt one token at a time ----
        past_key_values = None
        cur_attention_mask = None
        do_sample = gen_kwargs.get('do_sample', False)
        # temperature = gen_kwargs.get('temperature', 1.0)
        top_p = gen_kwargs.get('top_p', 1.0)
        with torch.no_grad():
            L = inputs.input_ids.size(1)
            for i in range(L):
                tok = inputs.input_ids[:, i:i+1]  # [1,1]
                if cur_attention_mask is None:
                    cur_attention_mask = inputs.attention_mask[:, i:i+1]
                else:
                    cur_attention_mask = torch.cat([cur_attention_mask, inputs.attention_mask[:, i:i+1]], dim=1)

                out = self.model(
                    input_ids=tok,
                    attention_mask=cur_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    **gen_kwargs
                )
                past_key_values = out.past_key_values

            # ---- Token-wise decoding ----
            generated: List[int] = []
            last_token = inputs.input_ids[:, -1:]  # start from last prompt token

            for _ in range(max_new_tokens):
                out = self.model(
                    input_ids=last_token,
                    attention_mask=cur_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    **gen_kwargs
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :]  # [1, V]

                if do_sample:
                    if temperature <= 0:
                        temperature = 1.0
                    probs = torch.softmax(logits / temperature, dim=-1)

                    if top_p < 1.0:
                        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                        cum = torch.cumsum(sorted_probs, dim=-1)
                        cutoff = cum > top_p
                        cutoff[..., 0] = False
                        sorted_probs[cutoff] = 0.0
                        sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)
                        next_id = sorted_idx.gather(-1, torch.multinomial(sorted_probs, num_samples=1))
                    else:
                        next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)  # [1,1]

                next_token_id = int(next_id.item())
                generated.append(next_token_id)

                # stop if eos/terminator
                if next_token_id in terminators:
                    break

                # update attention mask: append 1
                cur_attention_mask = torch.cat(
                    [cur_attention_mask, torch.ones((1, 1), device=self.device, dtype=cur_attention_mask.dtype)],
                    dim=1
                )
                last_token = next_id  # feed next token


        
        # Remove all hooks
        for h in handles:
            h.remove()
        
        gen_ids = torch.tensor(generated, device=self.device, dtype=torch.long)
        generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        return generated_text, len(generated)
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
        # import pdb; pdb.set_trace()
        # get <END> 的 id (已在__init__中添加)
        # end_token_id = self.tokenizer.convert_tokens_to_ids("<END>")
        stop_tokens = ["<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end▁of▁sentence|>"]
        terminators = [self.tokenizer.eos_token_id]  # 默认包含通用 eos
        # import pdb; pdb.set_trace()
        for token_text in stop_tokens:
            t_id = self.tokenizer.convert_tokens_to_ids(token_text)
            if t_id is not None:
                terminators.append(t_id)

        # 去重，防止重复 ID
        terminators = list(set(terminators))

        # terminators = [
        #     self.tokenizer.eos_token_id,
        #     self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]
        # Set default generation parameters
        gen_kwargs = {
            'do_sample': False,
            'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            # 'stopping_criteria': self._get_stopping_criteria(end_token_id, len(inputs.input_ids[0])),
            # 'eos_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': terminators,
            'output_hidden_states': True,
            'return_dict_in_generate': True,
            'max_new_tokens': max_new_tokens,
            'disable_compile': True,
            **generation_kwargs
        }
        # import pdb; pdb.set_trace()
        # 3. 准备存每一层、每一步的激活： {layer_name: [step0_vec, step1_vec, ...]}
        activations = {}
        handles = []

        # 4. 注册 forward hook 到指定层
        for layer_spec in hook_layers:
            hook_name = layer_spec
            activations[hook_name] = []
            # import pdb; pdb.set_trace()
            if hasattr(self.model.base_model, "language_model"):
                module = self.model.base_model.language_model.get_submodule(hook_name)
            else:
                module = self.model.base_model.get_submodule(hook_name)

            def make_hook(name):
                def hook_fn(module, inputs, outputs):
                    # import pdb; pdb.set_trace()
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    # 取最后一个 token 的隐藏状态
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
        # import pdb; pdb.set_trace()

        return generated_text, hidden_states, num_generated_tokens
       
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        **generation_kwargs
    ) -> Tuple[str, List[torch.Tensor]]:
        
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.device)

        stop_tokens = ["<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end▁of▁sentence|>"]
        terminators = [self.tokenizer.eos_token_id]  # 默认包含通用 eos

        for token_text in stop_tokens:
            t_id = self.tokenizer.convert_tokens_to_ids(token_text)
            if t_id is not None:
                terminators.append(t_id)

        terminators = list(set(terminators))

        gen_kwargs = {
            'do_sample': False,
            'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            'eos_token_id': terminators,
            'output_hidden_states': True,
            'return_dict_in_generate': True,
            'max_new_tokens': max_new_tokens,
            'disable_compile': True,
            **generation_kwargs
        }
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **gen_kwargs
            )


        # 7. 解码生成文本
        generated_ids = outputs.sequences[0][len(inputs.input_ids[0]):]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        num_generated_tokens = len(generated_ids)
     
        # 9. 清理显存
        del outputs
        del inputs
        del generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # import pdb; pdb.set_trace()

        return generated_text, num_generated_tokens
    
    
    def generate_dense_steered_response(
        self,
        prompt: str,
        hook_layers: List[str],  # {layer_name: feature_indices}
        max_new_tokens: Optional[int] = None,
        alpha: float = 1.0,  # steering strength
        direction = None,  # steering direction vector
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

        stop_tokens = ["<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<end_of_turn>",  "<|end▁of▁sentence|>"]
        terminators = [self.tokenizer.eos_token_id]  # 默认包含通用 eos
        for token_text in stop_tokens:
            t_id = self.tokenizer.convert_tokens_to_ids(token_text)
            if t_id is not None:
                terminators.append(t_id)

        # 去重，防止重复 ID
        terminators = list(set(terminators))

        # Set default generation parameters
        gen_kwargs = {
            'do_sample': False,
            'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            # 'stopping_criteria': self._get_stopping_criteria(end_token_id, len(inputs.input_ids[0])),
            'eos_token_id': terminators,
            'output_hidden_states': True,
            'output_scores': True,
            'return_dict_in_generate': True,
            'max_new_tokens': max_new_tokens,
            **generation_kwargs
        }

        handles = []

        # Register forward hooks for steering.
        # We steer only once: the first prompt prefill forward, last token only.
        layer_name = hook_layers[0]
        if hasattr(self.model.base_model, "language_model"):
            module = self.model.base_model.language_model.get_submodule(layer_name)
        else:
            module = self.model.base_model.get_submodule(layer_name)

        # import pdb; pdb.set_trace()
        layer_direction =  direction

        def make_steering_hook(layer_direction, strength):
            remaining = 1
            def hook_fn(module, inputs, outputs):
                nonlocal remaining
                if remaining <= 0:
                    return outputs

                tuple_flag = isinstance(outputs, tuple)
                if tuple_flag:
                    hidden = outputs[0]
                else:
                    hidden = outputs

                # hidden: [batch, seq, hidden_dim]
                direction_t = layer_direction.to(device=hidden.device, dtype=hidden.dtype)
                if direction_t.dim() == 1:
                    direction_t = direction_t.unsqueeze(0)  # [1, hidden_dim]
                # import pdb; pdb.set_trace()
                hidden[:, -1, :] = hidden[:, -1, :] + strength * direction_t
                remaining -= 1

                if tuple_flag:
                    return (hidden, *outputs[1:])
                return hidden
            return hook_fn

        h = module.register_forward_hook(make_steering_hook(layer_direction, alpha))
        handles.append(h)
        
        # Generate with steering
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **gen_kwargs
            )

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


    
    def _parse_response(self, response: str) -> str:
        """Parse response by removing <END> and everything after it.
        
        Args:
            response: Raw generated response
            
        Returns:
            Cleaned response without <|eot_id|> token
        """
        if '<|eot_id|>' in response:
            return response.split('<|eot_id|>')[0].strip()
        if '<end_of_turn>' in response:
            return response.split('<end_of_turn>')[0].strip()
        if '<eos>' in response:
            return response.split('<eos>')[0].strip()
        return response.strip()

    @property
    def name(self) -> str:
        """Get strategy name."""
        return self.__class__.__name__.replace('Strategy', '').lower()
