"""Simplified experiment runner for strategy comparison and activation analysis."""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import argparse
from utils.config import Config
from utils.random_utils import setup_reproducibility
from strategies.direct import DirectStrategy
from strategies.cot import ChainOfThoughtStrategy
from strategies.hint import HintStrategy
from sparsify import Sae
from analysis.LatentAnalyzer import LatentAnalyzer
import numpy as np
from analysis.SparseAutoEncoder import load_sae, SparseAutoEncoder
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from analysis.JumpReLUSAE import JumpReLUSAE

class ExperimentRunner:
    """Simplified runner for strategy comparison experiments."""
    
    def __init__(self, 
                config_path: str, 
                args: Optional[argparse.Namespace] = None,
                data_loader=None
                ):
        """Initialize experiment runner.
        
        Args:
            config_path: Path to configuration file
            args: Optional dictionary of additional arguments
        """
        self.config = Config(config_path)
        self.device = args.device
        self.args = args
        self.data_loader = data_loader
        # Set up reproducibility
        setup_reproducibility(self.config)
        
        # Initialize model and tokenizer
        self._load_model()
        self.load_sae()
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Setup output directory
        self.output_dir = Path(self.config.get('experiment.output_dir', './results') + "/" + self.config.get('model.name'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    
    def _load_model(self):
        """Load model and tokenizer."""
        transformers.logging.set_verbosity_error()
        model_name = self.config.get('model.name', 'gpt2-medium')
        # max_memory = {0: "0GiB", 1: "0GiB", 2: "0GiB", 3: "20GiB", 4: "20GiB", 5: "20GiB", 6: "20GiB", 7: "20GiB"}
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            torch_dtype=torch.float32 if model_name == "google/gemma-3-4b-it" else torch.float16,
            # device_map="auto",
            # max_memory=max_memory,
            # local_files_only=True
        ).to(self.device)
        
        self.model.eval()


    def load_sae(self) -> None:
        """Load SAE model"""
        print(f"Loading SAEs for layers {self.args.hook_layers}...")
        
        # Initialize as dictionary, not list
        self.saes = {}
        sae_model_name = self.config.get('sae.model_name', '')
        hook_layers = self.args.hook_layers if self.args.hook_layers else self.args.steer_layers
        for hook_layer in hook_layers:
            print(f"Loading SAE for {hook_layer}...")
            if sae_model_name.startswith("EleutherAI/"):
                # Load from Hugging Face Hub
                sae = Sae.load_from_hub(sae_model_name, hookpoint=hook_layer)
                
            elif sae_model_name.startswith("Goodfire/"):
                file_path = hf_hub_download(
                    repo_id=sae_model_name,
                    filename=f"{sae_model_name.split('/')[-1]}.pth" if sae_model_name=="Goodfire/Llama-3.1-8B-Instruct-SAE-l19" else f"{sae_model_name.split('/')[-1]}.pt",
                    repo_type="model"
                )

                sae = load_sae(
                    file_path,
                    d_model=self.model.config.hidden_size,
                    expansion_factor= 16 if sae_model_name=="Goodfire/Llama-3.1-8B-Instruct-SAE-l19" else 8,
                    device=self.device,
                )
            elif sae_model_name.startswith("google/"):
                Layer = hook_layer.split('.')[-1]
                LAYER = int(Layer)
                # import pdb; pdb.set_trace()
                path_to_params = hf_hub_download(
                    repo_id=self.config.get('sae.model_name'),
                    filename=f"resid_post/layer_{LAYER}_width_65k_l0_medium/params.safetensors",
                )
                params = load_file(path_to_params)
                d_model, d_sae = params["w_enc"].shape
                sae = JumpReLUSAE(d_model, d_sae)
                sae.load_state_dict(params)
                sae.cuda()
            elif sae_model_name.startswith("/"):
                sae = Sae.load_from_disk(os.path.join(sae_model_name, hook_layer))
            else:
                raise ValueError(f"Unsupported SAE model name: {sae_model_name}")
            self.saes[hook_layer] = sae
            print(f"SAE for layer {hook_layer} loaded.")
        
        print(f"Successfully loaded {len(self.saes)} SAE models")
    

    def _initialize_strategies(self):
        """Initialize reasoning strategies."""
        direct_model_config = {
            'max_new_tokens': self.config.get('strategies.direct.max_new_tokens', 128),
            'prompt_template': self.config.get('strategies.direct.prompt_template', None)
        }
        
        cot_model_config = {
            'max_new_tokens': self.config.get('strategies.cot.max_new_tokens', 256),
            'prompt_template': self.config.get('strategies.cot.prompt_template', None)
        }
        hint_model_config = {
            'max_new_tokens': self.config.get('strategies.hint.max_new_tokens', 256),
            'prompt_template': self.config.get('strategies.hint.prompt_template', None) 
        }
        self.strategies = {
            'direct': DirectStrategy(self.model, self.tokenizer, direct_model_config),
            'cot': ChainOfThoughtStrategy(self.model, self.tokenizer, cot_model_config),
            'hint': HintStrategy(self.model, self.tokenizer, hint_model_config),
        }
        
        print(f"Initialized {len(self.strategies)} strategies: {list(self.strategies.keys())}")
        
    def extract_latent_zs(
        self,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run experiment with all strategies on given question-answer pairs.
        
        Args:
            qa_pairs: List of question-answer pairs
            save_activations: Whether to save activation data to files
            
        Returns:
            Dict of strategy_name -> list of results
        """
        results = {}
        qa_pairs = self.data_loader.load_data(split=self.config.get('dataset.split', 'train'))
        print(f"Starting experiment with {len(qa_pairs)} question-answer pairs")
        
        # Save questions and indices first
        self._save_questions(qa_pairs)
        
        for strategy_name, strategy in self.strategies.items():
            layer_latent_zs = {}
            if self.config.get(f'strategies.{strategy_name}.skip', False):
                print(f"Skipping strategy: {strategy_name}")
                continue
            print(f"Running {strategy_name} strategy...")
            strategy_results = []
            
            for i, qa_pair in enumerate(tqdm(qa_pairs, desc=f"{strategy_name}")):
                question = qa_pair['question']
                ground_truth = qa_pair['answer']
                
                # Execute strategy
                if strategy_name == 'hint' or strategy_name == 'cot_hint':
                    answer = self.data_loader.extract_answer_for_hint(ground_truth)
                    output = strategy.execute(question, answer, self.args.hook_layers)
                else:
                    output = strategy.execute(question, self.args.hook_layers)
                
                
                latent_zs = self.get_latent_z(output.hidden_states)
                # import pdb; pdb.set_trace()
                for layer, z in zip(self.args.hook_layers, latent_zs):
                    layer_latent_zs.setdefault(layer, []).append(z)

                predicted_answer = output.metadata.get('answer', '')
                response = output.response
                num_generated_tokens = output.num_generated_tokens
                predicted_answer = self.data_loader.extract_answer(predicted_answer)
                result = {
                    'question_idx': i,
                    'question': question,
                    'response': response,
                    'num_generated_tokens': num_generated_tokens,
                    'predicted_answer': predicted_answer,
                    'ground_truth': ground_truth,
                    'correct': self.data_loader.check_answer_correctness(predicted_answer, ground_truth)
                }
                
                strategy_results.append(result)
                
            results[strategy_name] = strategy_results
            
            self._save_strategy_answers(strategy_results, strategy_name)

            for layer, zs in layer_latent_zs.items(): # 要不要把question_idx也存上
                layer_dir = self.output_dir / "latent_z" / self.config.get('dataset.name') / layer
                layer_dir.mkdir(parents=True, exist_ok=True)
                if self.args.type_of_analysis:
                    filename = f"{strategy_name}_latentz_{self.args.type_of_analysis}_{self.config.get('dataset.max_samples')}.pkl"
                else:
                    filename = f"{strategy_name}_latentz_tokenpos{self.args.token_pos}_{self.config.get('dataset.max_samples')}.pkl"
                filepath = layer_dir / filename
                with open(filepath, 'wb') as f:
                    pickle.dump(zs, f)
                print(f"Latent zs for {layer} and strategy {strategy_name} saved to {filepath}")


        return results
    
    def load_features(self):
        self.features={}
        type = self.args.get_index_type
        for layer in self.args.steer_layers:
            layer_dir = self.output_dir / "features" / self.config.get('dataset.name') / layer
            layer_dir.mkdir(parents=True, exist_ok=True)
            if self.args.type_of_analysis:
                filename = f"{type}_Features_{self.args.type_of_analysis}_TopK{self.args.topk}.npy"
            else:
                filename = f"{type}_Features_tokenpos{self.args.token_pos}_TopK{self.args.topk}.npy"
            filepath = layer_dir / filename
            features = np.load(filepath, allow_pickle=True)
            self.features[layer] = features
            print(f"features for {layer} loaded from {filepath}")


    def run_steering_experiment(
        self
    ) -> None:
        """Run steering experiment for a given strategy.
        
        Args:
            strategy_name: Name of the strategy to run steering on
        """
        target_strategy = self.args.steering_target_strategy
        if target_strategy not in self.strategies:
            print(f"Strategy '{target_strategy}' not found.")
            return
        
        strategy = self.strategies[target_strategy]
        self.load_features()
        if not self.features:
            raise ValueError("No features loaded, cannot proceed with steering.")
            
        
        qa_pairs = self.data_loader.load_data(split=self.config.get('dataset.split', 'train'))
        print(f"Starting steering experiment with {len(qa_pairs)} question-answer pairs using strategy '{target_strategy}'")
        results = []
        for i, qa_pair in enumerate(tqdm(qa_pairs, desc=f"Steering-{target_strategy}")):
            question = qa_pair['question']
            ground_truth = qa_pair['answer']
            output = strategy.steer(question = question, hook_layers_idx=self.features, k_index=self.args.k_index, saes=self.saes, alpha=self.args.steer_alpha, steer_n_steps=self.args.steer_n_steps)
                
            predicted_answer = output.metadata.get('answer', '')
            response = output.response
            num_generated_tokens = output.num_generated_tokens
            predicted_answer = self.data_loader.extract_answer(predicted_answer)
            result = {
                'question_idx': i,
                'question': question,
                'response': response,
                'num_generated_tokens': num_generated_tokens,
                'predicted_answer': predicted_answer,
                'ground_truth': ground_truth,
                'correct': self.data_loader.check_answer_correctness(predicted_answer, ground_truth)
            }
            
            results.append(result)
                
            
        self._save_strategy_answers(results, target_strategy+"_steered_"+self.args.get_index_type)

    def get_latent_z(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get latent representations from SAE models."""

        result_zs = []
        for sae, hidden_state in zip(self.saes.items(), hidden_states):
            hookpoint, sae = sae
            # print(f"Processing hidden state from {hookpoint} with SAE...")
            if hasattr(self.model, "hf_device_map"):
                # 找到该层对应的设备，例如 "cuda:1"
                target_device = self.model.hf_device_map.get(hookpoint, self.device)
            else:
                target_device = self.device
            hidden_state = hidden_state.to(target_device)
            sae = sae.to(target_device)
            sae_dtype = next(sae.parameters()).dtype  # SAE 当前用的 dtype
            hidden_state = hidden_state.to(sae_dtype)
            # import pdb; pdb.set_trace()
            latent_z = sae.encode(hidden_state)
            if sae.__class__.__name__ == "SparseAutoEncoder": # Goodfire 
                z = latent_z[0]
            elif sae.__class__.__name__ == "JumpReLUSAE": # Gemma Scope
                z = latent_z[0]
            else:   
                z = latent_z[2]  # Top-k activations
            if self.args.type_of_analysis == 'avg_pooling':
                z = torch.mean(z, dim=0)
            elif self.args.type_of_analysis == 'max_pooling':
                z, _ = torch.max(z, dim=0)
            else:
                z=z[self.args.token_pos]
            # import pdb; pdb.set_trace()
            result_zs.append(z.detach().cpu())
            # 清理GPU内存：移除中间变量和tensor
            sae = sae.cpu()  # 将SAE移回CPU释放GPU内存
            del hidden_state, z, latent_z
            torch.cuda.empty_cache()
        return result_zs
        
    

    
    def _save_questions(
        self, 
        qa_pairs: List[Dict[str, str]]
    ) -> None:
        """Save questions and ground truth answers to file.
        
        Args:
            qa_pairs: List of question-answer pairs
            timestamp: Experiment timestamp
        """
        questions_data = {
            'questions': [
                {'question_idx': i, 'question': qa['question'], 'ground_truth': qa['answer']} 
                for i, qa in enumerate(qa_pairs)
            ]
        }
        
        filepath = self.output_dir / "questions" / f"questions_{self.config.get('dataset.name')}_{self.config.get('dataset.max_samples')}.json"
        os.makedirs(filepath.parent, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(questions_data, f, indent=2)
        
        print(f"Questions saved to {filepath}")
    
    def _save_strategy_answers(
        self, 
        strategy_results: List[Dict[str, Any]], 
        strategy_name: str
    ) -> None:
        """Save complete results for a strategy including accuracy.
        
        Args:
            strategy_results: List of results with predictions and ground truth
            strategy_name: Strategy name
            timestamp: Experiment timestamp
        """
        # Calculate accuracy
        correct_count = sum(1 for r in strategy_results if r.get('correct', False))
        accuracy = correct_count / len(strategy_results) if strategy_results else 0.0

        # Calculate mean num_generated_tokens
        num_tokens_list = [r.get('num_generated_tokens', 0) for r in strategy_results if 'num_generated_tokens' in r]
        mean_num_generated_tokens = float(np.mean(num_tokens_list)) if num_tokens_list else 0.0
        std_num_generated_tokens = float(np.std(num_tokens_list)) if num_tokens_list else 0.0

        answers_data = {
            'strategy': strategy_name,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': len(strategy_results),
            'mean_num_generated_tokens': mean_num_generated_tokens,
            'std_num_generated_tokens': std_num_generated_tokens,
            'results': strategy_results
        }
        if strategy_name.endswith("_steered_"+self.args.get_index_type):
            if self.args.k_index is None:
                if self.args.type_of_analysis:
                    filepath = self.output_dir / "answers" / self.config.get('dataset.name') / self.args.steer_layers[0] / f"{strategy_name}_{self.args.type_of_analysis}_nsteps{self.args.steer_n_steps}_alpha{self.args.steer_alpha}_TopK{self.args.topk}_{self.config.get('dataset.max_samples')}.json"
                else:
                    filepath = self.output_dir / "answers" / self.config.get('dataset.name') / self.args.steer_layers[0] / f"{strategy_name}_tokenpos{self.args.token_pos}_nsteps{self.args.steer_n_steps}_alpha{self.args.steer_alpha}_TopK{self.args.topk}_{self.config.get('dataset.max_samples')}.json"
            else:
                first_layer = self.args.steer_layers[0]
                _, feaature_idx = self.features[first_layer]
                index = feaature_idx[self.args.k_index]
                if self.args.type_of_analysis:
                    filepath = self.output_dir / "answers" / self.config.get('dataset.name') / self.args.steer_layers[0] / f"{strategy_name}_{self.args.type_of_analysis}_nsteps{self.args.steer_n_steps}_alpha{self.args.steer_alpha}_featureidx{index}_{self.config.get('dataset.max_samples')}.json"
                else:
                    filepath = self.output_dir / "answers" / self.config.get('dataset.name') / self.args.steer_layers[0] / f"{strategy_name}_tokenpos{self.args.token_pos}_nsteps{self.args.steer_n_steps}_alpha{self.args.steer_alpha}_featureidx{index}_{self.config.get('dataset.max_samples')}.json"
        else:
            filepath = self.output_dir / "answers" / self.config.get('dataset.name') / f"{strategy_name}_{self.config.get('dataset.max_samples')}.json"
        os.makedirs(filepath.parent, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(answers_data, f, indent=2)
        
        print(f"Results for {strategy_name} saved to {filepath} (Accuracy: {accuracy:.2%})")
        
    
    def _tensors_to_numpy(self, obj: Any) -> Any:
        """Convert PyTorch tensors to numpy arrays recursively.
        
        Args:
            obj: Object potentially containing tensors
            
        Returns:
            Object with tensors converted to numpy arrays
        """
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        elif isinstance(obj, dict):
            return {k: self._tensors_to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._tensors_to_numpy(item) for item in obj]
        else:
            return obj


        

        