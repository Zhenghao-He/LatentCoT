"""Latent representation analyzer for comparing different strategies."""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import pickle


class LatentAnalyzer:
    """Analyzer for latent representations from different strategies."""
    
    def __init__(self, config, args):
        
        self.config = config
        self.args = args
        self.latents: Dict[str, Dict[str, Any]] = {}
        self.output_dir = Path(self.config.get('experiment.output_dir', './results') + "/" + self.config.get('model.name'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_latents(self):
        """Load latent representations for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy (e.g., 'direct', 'cot', 'hint')
            latent_file: Path to the latent representation file
        """
        strategies_config = self.config.get('strategies', {})
        for strategy_name, _ in strategies_config.items():
            if self.config.get(f'strategies.{strategy_name}.skip', False):
                print(f"Skipping loading latents for strategy '{strategy_name}' as per config.")
                continue
            for layer in self.args.hook_layers:
                layer_dir = self.output_dir / "latent_z" / self.config.get('dataset.name') / layer
                layer_dir.mkdir(parents=True, exist_ok=True)
                if self.args.type_of_analysis:
                    filename = f"{strategy_name}_latentz_{self.args.type_of_analysis}_{self.config.get('dataset.max_samples')}.pkl"
                else:
                    filename = f"{strategy_name}_latentz_tokenpos{self.args.token_pos}_{self.config.get('dataset.max_samples')}.pkl"
                filepath = layer_dir / filename

                with open(filepath, 'rb') as f:
                    zs = pickle.load(f)
                self.latents.setdefault(strategy_name, {})[layer] = zs
                print(f"Loaded latents for strategy '{strategy_name}', layer '{layer}' from {filepath}")
        return
        
    def compute_direction_by_difference(self, type:str) -> Optional[np.ndarray]:
        """Compute average direction vector from strategy A to strategy B.
        
        Args:
            strategy_a: Name of the first strategy
            strategy_b: Name of the second strategy
        """
        directions = {}
        if type == 'Reason':
            base_strategy = 'direct'
            target_strategy = 'cot'
        elif type == 'Hint':
            base_strategy = 'direct'
            target_strategy = 'hint'
        else:
            raise ValueError(f"Unknown analysis type: {type}")
        for layer in self.args.hook_layers:
            
            zs_a = self.latents.get(base_strategy, {}).get(layer) # (num_samples, latent_dim)
            zs_b = self.latents.get(target_strategy, {}).get(layer) # (num_samples, latent_dim)
            if zs_a is None or zs_b is None:
                print(f"Latents for strategies '{base_strategy}' or '{target_strategy}' not found for layer '{layer}'")
                continue
            
            # 每个问题的激活对应作差求平均
            
            zs_a = self.to_numpy(zs_a)
            zs_b = self.to_numpy(zs_b)

            # zs_a, zs_b: (num_samples, latent_dim)
            # 对应元素相减得到每个样本的方向向量
            diffs = zs_b - zs_a  # (num_samples, latent_dim)

            # 在样本维度求平均，得到平均方向向量
            avg_direction = np.mean(diffs, axis=0)  # (latent_dim,)
            
            idx = np.argsort(avg_direction)[::-1][:self.args.topk]
            
            top_values = avg_direction[idx]

            directions[layer] = (top_values, idx)
            layer_dir = self.output_dir / "features" / self.config.get('dataset.name') / layer
            layer_dir.mkdir(parents=True, exist_ok=True)
            if self.args.type_of_analysis:
                filename = f"{type}_Features_{self.args.type_of_analysis}_TopK{self.args.topk}.npy"
            else:
                filename = f"{type}_Features_tokenpos{self.args.token_pos}_TopK{self.args.topk}.npy"
            filepath = layer_dir / filename
            np.save(filepath, (top_values, idx))
            print(f"acts and index for {layer} saved to {filepath}")
        
            # import pdb; pdb.set_trace() 

    def to_numpy(self, data):
        """Convert data to numpy array, handling PyTorch tensors and BFloat16."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().float().numpy()  # 强制转 float32
        elif isinstance(data, list):
            return np.array([self.to_numpy(item) for item in data])
        else:
            return np.array(data)

