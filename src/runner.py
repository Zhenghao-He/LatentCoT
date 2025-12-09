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
        self.output_dir = Path(self.config.get('experiment.output_dir', './results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    
    def _load_model(self):
        """Load model and tokenizer."""
        transformers.logging.set_verbosity_error()
        model_name = self.config.get('model.name', 'gpt2-medium')

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
            torch_dtype=torch.float32,
            # local_files_only=True
        ).to(self.device)
        
        self.model.eval()


    def load_sae(self) -> None:
        """Load SAE model"""
        print(f"Loading SAEs for layers {self.args.hook_layers}...")
        
        # Initialize as dictionary, not list
        self.saes = {}
        sae_model_name = self.config.get('sae.model_name', '')
        
        for hook_layer in self.args.hook_layers:
            print(f"Loading SAE for {hook_layer}...")
            
            sae = Sae.load_from_hub(sae_model_name, hookpoint=hook_layer)
            
            self.saes[hook_layer] = sae
            print(f"SAE for layer {hook_layer} loaded - input_dim={sae.d_in}, num_latents={sae.num_latents}")
        
        print(f"Successfully loaded {len(self.saes)} SAE models")
    

    def _initialize_strategies(self):
        """Initialize reasoning strategies."""
        direct_model_config = {
            'max_new_tokens': self.config.get('strategies.direct.max_new_tokens', 128)
        }
        
        cot_model_config = {
            'max_new_tokens': self.config.get('strategies.cot.max_new_tokens', 256)
        }
        hint_model_config = {
            'max_new_tokens': self.config.get('strategies.hint.max_new_tokens', 256)
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
        layer_latent_zs = {}
        for strategy_name, strategy in self.strategies.items():
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
                for layer, z in zip(self.args.hook_layers, latent_zs):
                    layer_latent_zs.setdefault(layer, []).append(z)

                predicted_answer = output.metadata.get('answer', '')
                
                result = {
                    'question_idx': i,
                    'question': question,
                    'predicted_answer': self.data_loader.extract_answer(predicted_answer),
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
    
    def get_latent_z(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get latent representations from SAE models."""

        result_zs = []
        for sae, hidden_state in zip(self.saes.items(), hidden_states):
            hookpoint, sae = sae
            # print(f"Processing hidden state from {hookpoint} with SAE...")
            hidden_state = hidden_state.to(self.device)
            sae = sae.to(self.device)
            sae_dtype = next(sae.parameters()).dtype  # SAE 当前用的 dtype
            hidden_state = hidden_state.to(sae_dtype)
            # import pdb; pdb.set_trace()
            latent_z = sae.encode(hidden_state)
            if self.args.type_of_analysis == 'avg_pooling':
                z=latent_z[2]
                z = torch.mean(z, dim=0)
            elif self.args.type_of_analysis == 'max_pooling':
                z=latent_z[2]
                z, _ = torch.max(z, dim=0)
            else:
                z=latent_z[2]
                z=z[self.args.token_pos]
            # import pdb; pdb.set_trace()
            result_zs.append(z.cpu())
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
        
        answers_data = {
            'strategy': strategy_name,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': len(strategy_results),
            'results': strategy_results
        }
        
        filepath = self.output_dir / "answers" / f"{strategy_name}_answers_{self.config.get('dataset.name')}_{self.config.get('dataset.max_samples')}.json"
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

    def run_analysis(self):
        """Run latent representation analysis."""
        
        analyzer = LatentAnalyzer(self.config, self.args)
        analyzer.load_latents()
        
        # Example: compute direction between 'direct' and 'cot'
        strategy_a = 'direct'
        strategy_b = 'cot'
        direction = analyzer.compute_direction(strategy_a, strategy_b)
        if direction is not None:
            print(f"Computed direction from {strategy_a} to {strategy_b}: shape {direction.shape}")
        else:
            print(f"Could not compute direction between {strategy_a} and {strategy_b}")