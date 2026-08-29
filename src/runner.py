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
from utils.tools import add_with_zero_pad
from scipy.stats import pointbiserialr
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
        # Setup output directory
        self.output_dir = Path(self.config.get('experiment.output_dir', './results') + "/" + self.config.get('model.name'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # features = self.load_features()
        # import pdb; pdb.set_trace()

        # Initialize model and tokenizer
        self._load_model()
        self.load_sae()
        
        # Initialize strategies
        self._initialize_strategies()
        
        
        
    
    def _load_model(self):
        """Load model and tokenizer."""
        transformers.logging.set_verbosity_error()
        model_name = self.config.get('model.name', 'gpt2-medium')
        # max_memory = {0: "0GiB", 1: "81GiB", 2: "81GiB", 3: "0GiB"}
        # max_memory = {0: "15GiB", 1: "15GiB", 2: "30GiB", 3: "48GiB"}
        max_memory = {0: "0GiB", 1: "0GiB", 2: "0GiB", 3: "48GiB", 4: "48GiB", 5: "48GiB", 6: "48GiB", 7: "48GiB"}
        # max_memory = {0: "0GiB", 1: "0GiB", 2: "0GiB", 3: "48GiB", 4: "48GiB", 5: "0GiB", 6: "0GiB", 7: "0GiB"}
        # max_memory = {0: "0GiB", 1: "48GiB", 2: "48GiB", 3: "0GiB", 4: "0GiB", 5: "0GiB", 6: "0GiB", 7: "0GiB"}
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.args.multi_gpu:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                output_hidden_states=True,
                torch_dtype=torch.float32 if model_name.startswith("google/") else torch.float16,
                device_map="auto",
                max_memory=max_memory,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                output_hidden_states=True,
                torch_dtype=torch.float32 if model_name.startswith("google/") else torch.float16,
            ).to(self.device)
        
        self.model.eval()


    def load_sae(self) -> None:
        """Load SAE model"""
        print(f"Loading SAEs for layers {self.args.hook_layers}...")
        
        # Initialize as dictionary, not list
        self.saes = {}
        sae_model_name = self.config.get('sae.model_name', '')
        hook_layers = self.args.hook_layers if self.args.hook_layers else self.args.steer_layers
        if hook_layers is None:
            print("No hook layers specified for SAE loading.")
            return
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
                    filename=f"resid_post/layer_{LAYER}_width_262k_l0_medium/params.safetensors",
                )
                params = load_file(path_to_params)
                d_model, d_sae = params["w_enc"].shape
                sae = JumpReLUSAE(d_model, d_sae)
                sae.load_state_dict(params)
                # sae.cuda()
                # import pdb; pdb.set_trace()
                hidden_device = next(self.model.language_model.get_submodule(hook_layer).parameters()).device
                # sae.to(hidden_device)
                # sae.to(self.device)
            elif sae_model_name.startswith("andreuka18/"):
                from sae_lens import SAE

                sae, cfg_dict, sparsity = SAE.from_pretrained("andreuka18/sae-deepseek-r1-llama-8b", f"model.{hook_layer}")
                # import pdb; pdb.set_trace() 

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
        think_model_config = {
            'max_new_tokens': self.config.get('strategies.think.max_new_tokens', 256),
            'prompt_template': self.config.get('strategies.think.prompt_template', None)
        }
        solve_model_config = {  
            'max_new_tokens': self.config.get('strategies.solve.max_new_tokens', 256),
            'prompt_template': self.config.get('strategies.solve.prompt_template', None)
        }
        explain_model_config = {
            'max_new_tokens': self.config.get('strategies.explain.max_new_tokens', 256),
            'prompt_template': self.config.get('strategies.explain.prompt_template', None)
        }
        pseudo_reason_model_config = {
            'max_new_tokens': self.config.get('strategies.pseudo_reason.max_new_tokens', 256),
            'prompt_template': self.config.get('strategies.pseudo_reason.prompt_template', None)
        }
        self.strategies = {
            'direct': DirectStrategy(self.model, self.tokenizer, direct_model_config),
            'cot': ChainOfThoughtStrategy(self.model, self.tokenizer, cot_model_config),
            'hint': HintStrategy(self.model, self.tokenizer, hint_model_config),
            'think': ChainOfThoughtStrategy(self.model, self.tokenizer, think_model_config),
            'solve': ChainOfThoughtStrategy(self.model, self.tokenizer, solve_model_config),
            'explain': ChainOfThoughtStrategy(self.model, self.tokenizer, explain_model_config),
            'pseudo_reason': ChainOfThoughtStrategy(self.model, self.tokenizer, pseudo_reason_model_config)
        }
        
        print(f"Initialized {len(self.strategies)} strategies: {list(self.strategies.keys())}")
    
    def get_raw_activations(
        self,
    ) -> Dict[str, List[torch.Tensor]]:
        """Get raw activations for a given question and strategy."""

        qa_pairs = self.data_loader.load_data(split=self.config.get('dataset.split', 'train'))
        print(f"Starting experiment with {len(qa_pairs)} question-answer pairs")
        
        for strategy_name, strategy in self.strategies.items():
            layer_latent_acts = {}
            if self.config.get(f'strategies.{strategy_name}.skip', False):
                print(f"Skipping strategy: {strategy_name}")
                continue
            print(f"Running {strategy_name} strategy...")

            for i, qa_pair in enumerate(tqdm(qa_pairs, desc=f"{strategy_name}")):
                
                question = qa_pair['question']
                
                
                acts = strategy.get_raw_acts(question, self.args.hook_layers)

                for layer in self.args.hook_layers:
                    vals = acts[layer][0]
                    layer_latent_acts.setdefault(layer, []).append(vals)


                if (i+1) % 50 == 0:
                    self.save_raw_acts(strategy_name, layer_latent_acts)

            self.save_raw_acts(strategy_name, layer_latent_acts)

    def save_raw_acts(self, strategy_name, layer_latent_acts):
        for layer, acts in layer_latent_acts.items():
            layer_dir = self.output_dir / "raw_acts" / self.config.get('dataset.name') / layer
            layer_dir.mkdir(parents=True, exist_ok=True)
            if self.args.type_of_analysis:
                filename = f"{strategy_name}_raw_acts_{self.args.type_of_analysis}_{self.config.get('dataset.max_samples')}.pkl"
            else:
                filename = f"{strategy_name}_raw_acts_tokenpos{self.args.token_pos}_{self.config.get('dataset.max_samples')}.pkl"
            filepath = layer_dir / filename
            with open(filepath, 'wb') as f:
                pickle.dump(acts, f)
            print(f"Raw acts for {layer} and strategy {strategy_name} saved to {filepath}")

    def load_raw_acts(self, strategy_name, layer):
        layer_dir = self.output_dir / "raw_acts" / self.config.get('dataset.name') / layer
        layer_dir.mkdir(parents=True, exist_ok=True)
        if self.args.type_of_analysis:
            filename = f"{strategy_name}_raw_acts_{self.args.type_of_analysis}_{self.config.get('dataset.max_samples')}.pkl"
        else:
            filename = f"{strategy_name}_raw_acts_tokenpos{self.args.token_pos}_{self.config.get('dataset.max_samples')}.pkl"
        filepath = layer_dir / filename
        if not filepath.exists():
            self.get_raw_activations()
        with open(filepath, 'rb') as f:
            acts = pickle.load(f)
        print(f"Raw acts for {layer} and strategy {strategy_name} loaded from {filepath}")
        return acts

    def record_activations(
        self,
    ) -> None:
        """Record model activations for given question-answer pairs.
        
        Args:
            qa_pairs: List of question-answer pairs
        """
        
        self.load_features()

        qa_pairs = self.data_loader.load_data(split=self.config.get('dataset.split', 'train'))
        print(f"Starting experiment with {len(qa_pairs)} question-answer pairs")
        
        
        for strategy_name, strategy in self.strategies.items():
            layer_latent_acts = {}
            if self.config.get(f'strategies.{strategy_name}.skip', False):
                print(f"Skipping strategy: {strategy_name}")
                continue
            print(f"Running {strategy_name} strategy...")

            for i, qa_pair in enumerate(tqdm(qa_pairs, desc=f"{strategy_name}")):
                
                question = qa_pair['question']
                
                
                acts = strategy.get_activations(question, features=self.features, k_index=self.args.k_index, saes=self.saes)
                # import pdb; pdb.set_trace()

                for layer in self.args.steer_layers:
                    vals = torch.cat(acts[layer])
                    layer_latent_acts.setdefault(layer, []).append(vals)


                if (i+1) % 50 == 0:
                    self.save_latent_activations(strategy_name, layer_latent_acts)

            self.save_latent_activations(strategy_name, layer_latent_acts)
            ####save here#####

    def record_all_activations(
        self,
    ) -> None:
        
        self.load_features()

        qa_pairs = self.data_loader.load_data(split=self.config.get('dataset.split', 'train'))
        print(f"Starting experiment with {len(qa_pairs)} question-answer pairs")
        
        # strategy_name = 'cot'
        # strategy = self.strategies[strategy_name]
        for strategy_name, strategy in self.strategies.items():
            layer_latent_acts = {}
            if self.config.get(f'strategies.{strategy_name}.skip', False):
                print(f"Skipping strategy: {strategy_name}")
                continue
            print(f"Running {strategy_name} strategy...")

            for i, qa_pair in enumerate(tqdm(qa_pairs, desc=f"{strategy_name}")):
                
                question = qa_pair['question']
                
                
                acts = strategy.get_activations(question, features=self.features, k_index=self.args.k_index, saes=self.saes)

                for layer in self.args.steer_layers:
                    vals = torch.cat(acts[layer])
                    layer_latent_acts.setdefault(layer, []).append(vals)


                if (i+1) % 50 == 0:
                    self.save_latent_activations_all(strategy_name, layer_latent_acts)

            self.save_latent_activations_all(strategy_name, layer_latent_acts)

    
    def construct_dense_direction(self):
        # import pdb; pdb.set_trace()
        
        dense_direction = {}
        for layer in self.args.hook_layers:
            # direct_layer_acts = direct_acts[layer]
            # cot_layer_acts = cot_acts[layer]
            direct_acts = self.load_raw_acts('direct', self.args.hook_layers[0])
            cot_acts = self.load_raw_acts('cot', self.args.hook_layers[0])
            # import pdb; pdb.set_trace()
            direct_mean_act = torch.stack(direct_acts).mean(dim=0)
            cot_mean_act = torch.stack(cot_acts).mean(dim=0)
            direction = cot_mean_act - direct_mean_act
            direction = direction / (direction.norm(p=2) + 1e-12)
            dense_direction[layer] = direction

            layer_dir = self.output_dir / "dense_directions" / self.config.get('dataset.name') / layer
            layer_dir.mkdir(parents=True, exist_ok=True)
            filename = f"dense_direction_{self.config.get('dataset.max_samples')}.pkl"
            filepath = layer_dir / filename
            torch.save(direction, filepath)
            import pdb; pdb.set_trace() 
        return dense_direction

    def save_latent_activations_all(self,strategy_name, layer_latent_acts):
        for layer, acts in layer_latent_acts.items(): 
            # for i, act in enumerate(acts):
            #     acts[i] = acts[i][:self.args.max_activation_length]

            layer_dir = self.output_dir / "latent_acts_full" / self.config.get('dataset.name') / layer
            layer_dir.mkdir(parents=True, exist_ok=True)
            if self.args.type_of_analysis:
                filename = f"{strategy_name}_latent_acts_{self.args.type_of_analysis}_{self.config.get('dataset.max_samples')}.pkl"
            else:
                filename = f"{strategy_name}_latent_acts_tokenpos{self.args.token_pos}_{self.config.get('dataset.max_samples')}.pkl"
            filepath = layer_dir / filename
            with open(filepath, 'wb') as f:
                pickle.dump(acts, f)
            print(f"Latent acts for {layer} and strategy {strategy_name} saved to {filepath}")
        # import pdb; pdb.set_trace()

    def load_latent_activations_all(self, strategy_name, layer):
        layer_dir = self.output_dir / "latent_acts_full" / self.config.get('dataset.name') / layer
        layer_dir.mkdir(parents=True, exist_ok=True)
        if self.args.type_of_analysis:
            filename = f"{strategy_name}_latent_acts_{self.args.type_of_analysis}_{self.config.get('dataset.max_samples')}.pkl"
        else:
            filename = f"{strategy_name}_latent_acts_tokenpos{self.args.token_pos}_{self.config.get('dataset.max_samples')}.pkl"
        filepath = layer_dir / filename
        if not filepath.exists():
            self.record_all_activations()
        with open(filepath, 'rb') as f:
            acts = pickle.load(f)
        print(f"Latent acts for {layer} and strategy {strategy_name} loaded from {filepath}")
        return acts
    
    def eval_latent_activations(
        self,
    ):
        acts_all_strategy = {}
        for strategy_name, strategy in self.strategies.items():
            if self.config.get(f'strategies.{strategy_name}.skip', False):
                print(f"Skipping strategy: {strategy_name}")
                continue
            answers = self._load_strategy_answers(strategy_name)

            ####为了rebuttal 以后删掉#####
            # acts_all_strategy[strategy_name] = {}
            # layer_acts = self.load_latent_activations_all(strategy_name, self.args.steer_layers[0])
            # acts_all_strategy[strategy_name]= layer_acts
            # continue
            ######
            # import  pdb; pdb.set_trace()
            # steered_answers = self._load_strategy_answers(strategy_name +'_steered_'+self.args.get_index_type)
            # steered_answers = steered_answers['results']
            layer_acts_correct = {}
            layer_acts_incorrect = {}
            
            acts_all_strategy[strategy_name] = {}
            for layer in self.args.steer_layers:
                layer_acts = self.load_latent_activations_all(strategy_name, layer)
                # import pdb; pdb.set_trace()
                correct_acts = []
                incorrect_acts = []
                for i, act in enumerate(layer_acts):
                    layer_acts[i] = layer_acts[i][:self.args.max_activation_length]
                    # import pdb; pdb.set_trace()
                    if answers[i]['correct']==True:
                        correct_acts.append(layer_acts[i])
                    else:
                        incorrect_acts.append(layer_acts[i])
                acts_all_strategy[strategy_name][layer] = layer_acts
                layer_acts_correct[layer]=correct_acts
                layer_acts_incorrect[layer]=incorrect_acts

                correct_scores = np.array([act.max().item() for act in correct_acts])
                incorrect_scores = np.array([act.max().item() for act in incorrect_acts])
                print(f"Correct answers: mean max activation {correct_scores.mean()}, std {correct_scores.std()}")
                print(f"Incorrect answers: mean max activation {incorrect_scores.mean()}, std {incorrect_scores.std()}")

                scores = np.concatenate([
                    correct_scores,
                    incorrect_scores
                ])
                labels = np.concatenate([
                    np.ones(len(correct_scores)),     # 正确 = 1
                    np.zeros(len(incorrect_scores))   # 错误 = 0
                ])
                r, p = pointbiserialr(labels, scores)
                print("point-biserial r =", r)
                print("p-value =", p)
                save_dict = {
                    "mean_correct": float(correct_scores.mean()),
                    "mean_incorrect": float(incorrect_scores.mean()),
                    "r_pointbiserial": float(r),
                    "p_pointbiserial": float(p),
                    "n_correct": int(len(correct_scores)),
                    "n_incorrect": int(len(incorrect_scores)),
                }
                save_dir = self.output_dir / "latent_acts_full" / self.config.get('dataset.name') / layer
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / (strategy_name + "_reasoning_feature_stats.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump(save_dict, f)
            self.save_latent_activations_all(strategy_name + '_correct', layer_acts_correct)
            self.save_latent_activations_all(strategy_name + '_incorrect', layer_acts_incorrect)

        ##########
        # pseudo_acts = acts_all_strategy['pseudo_reason']
        # cot_acts = acts_all_strategy['cot']
        # avg_pseudo_acts = [act.mean().item() for act in pseudo_acts]
        # avg_cot_acts = [act.mean().item() for act in cot_acts]
        # direct_acts = acts_all_strategy['direct']
        # avg_direct_acts = [act.mean().item() for act in direct_acts]
        # max_pseudo_acts = [act.max().item() for act in pseudo_acts]
        # max_cot_acts = [act.max().item() for act in cot_acts]
        # max_direct_acts = [act.max().item() for act in direct_acts]
        # max_avg_pseudo_acts = np.array(max_pseudo_acts).mean().item()
        # import pdb; pdb.set_trace() 
        ############
        direct_acts = acts_all_strategy['direct']
        cot_acts = acts_all_strategy['cot']
        for layer in self.args.steer_layers:
            direct_layer_acts = direct_acts[layer]
            cot_layer_acts = cot_acts[layer]
            cot_scores = np.array([act.max().item() for act in cot_layer_acts])
            direct_scores = np.array([act.max().item() for act in direct_layer_acts])
            print("mean cot activation:", cot_scores.mean(), "std:", cot_scores.std())
            print("mean direct activation:", direct_scores.mean(), "std:", direct_scores.std())
            scores = np.concatenate([
                cot_scores,
                direct_scores
            ])
            labels = np.concatenate([
                np.ones(len(cot_scores)),     # cot = 1
                np.zeros(len(direct_scores))   # direct = 0
            ])
            r, p = pointbiserialr(labels, scores)
            print("point-biserial r between cot and direct =", r)
            print("p-value =", p)
            save_dict = {
                "mean_cot": float(cot_scores.mean()),
                "mean_direct": float(direct_scores.mean()),
                "r_pointbiserial": float(r),
                "p_pointbiserial": float(p),
                "n_cot": int(len(cot_scores)),
                "n_direct": int(len(direct_scores)),
            }
            save_dir = self.output_dir / "latent_acts_full" / self.config.get('dataset.name') / layer
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / ("cot_vs_direct_reasoning_feature_stats.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(save_dict, f)

    def save_latent_activations(self, strategy_name, layer_latent_acts):
        for layer, acts in layer_latent_acts.items():
            # acts: list of 1D tensors, variable length

            max_len = max(act.numel() for act in acts)

            sum_act = torch.zeros(max_len, dtype=acts[0].dtype)
            count_act = torch.zeros(max_len, dtype=torch.long)
            sq_sum_act = torch.zeros(max_len, dtype=acts[0].dtype)
            for act in acts:
                L = act.numel()
                sum_act[:L] += act
                count_act[:L] += 1
                sq_sum_act[:L] += act ** 2

            # avoid division by zero
            mean_act = sum_act / count_act.clamp(min=1)
            
            var = sq_sum_act / count_act - mean_act ** 2
            stderr = torch.sqrt(var) / torch.sqrt(count_act )

            layer_dir = self.output_dir / "latent_acts" / self.config.get('dataset.name') / layer
            layer_dir.mkdir(parents=True, exist_ok=True)

            if self.args.type_of_analysis:
                filename = f"{strategy_name}_latent_acts_{self.args.type_of_analysis}_{self.config.get('dataset.max_samples')}.pkl"
            else:
                filename = f"{strategy_name}_latent_acts_tokenpos{self.args.token_pos}_{self.config.get('dataset.max_samples')}.pkl"

            filepath = layer_dir / filename
            with open(filepath, "wb") as f:
                pickle.dump(
                    {
                        "mean": mean_act.cpu(),
                        "count": count_act.cpu(),
                        "stderr": stderr.cpu(),
                        "var": var.cpu()
                    },
                    f
                )

            print(f"Latent acts for {layer} and strategy {strategy_name} saved to {filepath}")

        
    def run_baseline(
        self,
    ) -> Dict[str, List[Dict[str, Any]]]:
        
        results = {}
        qa_pairs = self.data_loader.load_data(split=self.config.get('dataset.split', 'train'))
        print(f"Starting experiment with {len(qa_pairs)} question-answer pairs")
        
        # Save questions and indices first
        self._save_questions(qa_pairs)
        
        for strategy_name, strategy in self.strategies.items():
            # import pdb; pdb.set_trace() 
            if self.config.get(f'strategies.{strategy_name}.skip', False):
                print(f"Skipping strategy: {strategy_name}")
                continue
            print(f"Running {strategy_name} strategy...")
            strategy_results = []
            
            strategy_results = self._load_strategy_answers(strategy_name)
            
            
            if len(strategy_results) == len(qa_pairs):
                print(f"All results for {strategy_name} already computed, skipping...")
                results[strategy_name] = strategy_results
                continue

            for i, qa_pair in enumerate(tqdm(qa_pairs, desc=f"{strategy_name}")):
                if len(strategy_results) > i:
                    continue  # Skip already computed results
                question = qa_pair['question']
                ground_truth = qa_pair['answer']
                

                output = strategy.execute_baseline(question, self.args.hook_layers)
                

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
                if (i+1) % 10 == 0:
                    self._save_strategy_answers(strategy_results, strategy_name)

            results[strategy_name] = strategy_results
            
            self._save_strategy_answers(strategy_results, strategy_name)

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
            
            strategy_results = self._load_strategy_answers(strategy_name)
            layer_latent_zs, num_features = self._load_latent_zs(self.args.hook_layers, strategy_name)
            if len(strategy_results) != num_features:
                print(f"Mismatch in loaded results and latent zs for {strategy_name}, recomputing...")
                strategy_results = []
                layer_latent_zs = {}
            if len(strategy_results) == len(qa_pairs):
                print(f"All results for {strategy_name} already computed, skipping...")
                results[strategy_name] = strategy_results
                continue
            # import pdb; pdb.set_trace()   

            for i, qa_pair in enumerate(tqdm(qa_pairs, desc=f"{strategy_name}")):
                if len(strategy_results) > i:
                    continue  # Skip already computed results
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
                if (i+1) % 10 == 0:
                    self._save_strategy_answers(strategy_results, strategy_name)
                    self._save_latent_zs(layer_latent_zs, strategy_name)

            results[strategy_name] = strategy_results
            
            self._save_strategy_answers(strategy_results, strategy_name)
            self._save_latent_zs(layer_latent_zs, strategy_name)
            # for layer, zs in layer_latent_zs.items(): # 要不要把question_idx也存上
            #     layer_dir = self.output_dir / "latent_z" / self.config.get('dataset.name') / layer
            #     layer_dir.mkdir(parents=True, exist_ok=True)
            #     if self.args.type_of_analysis:
            #         filename = f"{strategy_name}_latentz_{self.args.type_of_analysis}_{self.config.get('dataset.max_samples')}.pkl"
            #     else:
            #         filename = f"{strategy_name}_latentz_tokenpos{self.args.token_pos}_{self.config.get('dataset.max_samples')}.pkl"
            #     filepath = layer_dir / filename
            #     with open(filepath, 'wb') as f:
            #         pickle.dump(zs, f)
            #     print(f"Latent zs for {layer} and strategy {strategy_name} saved to {filepath}")


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
        # results = self._load_strategy_answers(target_strategy+"_steered_"+self.args.get_index_type)
        for i, qa_pair in enumerate(tqdm(qa_pairs, desc=f"Steering-{target_strategy}")):
            # if len(results) > i:
            #     continue  # Skip already computed results
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
            if (i+1) % 10 == 0:
                self._save_strategy_answers(results, target_strategy+"_steered_"+self.args.get_index_type)
                
            
        self._save_strategy_answers(results, target_strategy+"_steered_"+self.args.get_index_type)

    def get_latent_z(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get latent representations from SAE models."""

        result_zs = []
        for sae, hidden_state in zip(self.saes.items(), hidden_states):
            hookpoint, sae = sae
            # print(f"Processing hidden state from {hookpoint} with SAE...")
            if hasattr(self.model, "hf_device_map"):
                # 找到该层对应的设备，例如 "cuda:1"
                if hasattr(self.model.base_model, "language_model"):
                    # layer_module = self.model.base_model.language_model.get_submodule(layer_name)
                    target_device = self.model.hf_device_map.get("model.language_model." + hookpoint, self.device)
                else:
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
                # import pdb; pdb.set_trace()
            elif sae.__class__.__name__ == "JumpReLUSAE": # Gemma Scope
                z = latent_z[0]
            elif sae.__class__.__name__ == "StandardSAE": 
                z = latent_z
                # import pdb; pdb.set_trace()
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
        
    
    def _save_latent_zs(
        self, 
        layer_latent_zs: Any,
        strategy_name: str
    ) -> None:
        """Save latent representations to file.
        
        Args:
            layer: Layer name
            strategy_name: Strategy name
            latent_zs: List of latent representations
            timestamp: Experiment timestamp
        """
        
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


    def _load_latent_zs(
        self, 
        layers: List[str],
        strategy_name: str
    ) -> List[Any]:
        """Load latent representations from file.
        
        Args:
            layer: Layer name
            strategy_name: Strategy name
            
        Returns:
            List of latent representations
        """
        layer_latent_zs = {}
        previous_num_features = 0
        for layer in layers:
            layer_dir = self.output_dir / "latent_z" / self.config.get('dataset.name') / layer
            if self.args.type_of_analysis:
                filename = f"{strategy_name}_latentz_{self.args.type_of_analysis}_{self.config.get('dataset.max_samples')}.pkl"
            else:
                filename = f"{strategy_name}_latentz_tokenpos{self.args.token_pos}_{self.config.get('dataset.max_samples')}.pkl"
            filepath = layer_dir / filename
            if not os.path.exists(filepath):
                print(f"No saved latent zs found for {layer} and strategy {strategy_name} at {filepath}")
                return {}, 0
            with open(filepath, 'rb') as f:
                latent_zs = pickle.load(f)
            # import pdb; pdb.set_trace()
            num_features = len(latent_zs)
            if previous_num_features != 0 and num_features != previous_num_features:
                raise ValueError(f"Mismatch in number of features for layer {layer}: expected {previous_num_features}, got {num_features}")
            previous_num_features = num_features
            layer_latent_zs[layer] = latent_zs
            print(f"Latent zs for {layer} and strategy {strategy_name} loaded from {filepath}")
        return layer_latent_zs, previous_num_features
    
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
    
    def _load_strategy_answers(
        self, 
        strategy_name: str
    ) -> List[Dict[str, Any]]:
        pass
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
        if not os.path.exists(filepath):
            print(f"No saved results found for {strategy_name} at {filepath}")
            return []
        with open(filepath, 'r') as f:
            answers_data = json.load(f)
        
        print(f"Results for {strategy_name} loaded from {filepath} (Accuracy: {answers_data.get('accuracy', 0):.2%})")
        return answers_data.get('results', [])

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


    def run_dense_steering_experiment(
        self
    ) -> None:
        """Run steering experiment for a given strategy.
        
        Args:
            strategy_name: Name of the strategy to run steering on
        """
        layer = self.args.hook_layers[0]
        layer_dir = self.output_dir / "dense_directions" / self.config.get('dataset.name') / layer
        layer_dir.mkdir(parents=True, exist_ok=True)
        filename = f"dense_direction_{self.config.get('dataset.max_samples')}.pkl"
        filepath = layer_dir / filename
        direction = torch.load(filepath)

        target_strategy = self.args.steering_target_strategy
        if target_strategy not in self.strategies:
            print(f"Strategy '{target_strategy}' not found.")
            return
        
        strategy = self.strategies[target_strategy]
        
        qa_pairs = self.data_loader.load_data(split=self.config.get('dataset.split', 'train'))
        print(f"Starting steering experiment with {len(qa_pairs)} question-answer pairs using strategy '{target_strategy}'")
        results = []
        # results = self._load_strategy_answers(target_strategy+"_steered_"+self.args.get_index_type)
        for i, qa_pair in enumerate(tqdm(qa_pairs, desc=f"Steering-{target_strategy}")):
            # if len(results) > i:
            #     continue  # Skip already computed results
            question = qa_pair['question']
            ground_truth = qa_pair['answer']
            output = strategy.dense_steer(question = question, hook_layers=self.args.hook_layers, alpha=self.args.steer_alpha, direction=direction)
                
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
            if (i+1) % 50 == 0:
                self._save_strategy_answers(results, target_strategy+"_dense_steered_")
                
            
        self._save_strategy_answers(results, target_strategy+"_dense_steered_")


        
