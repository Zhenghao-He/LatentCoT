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
        
    def load_latents(self):
        """Load latent representations for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy (e.g., 'direct', 'cot', 'hint')
            latent_file: Path to the latent representation file
        """
        # strategies = ['direct', 'cot', 'hint'] #暂时默认这3种
        for strategy in self.config.get('strategies', []):
            strategy_name = strategy['name']
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
        
    def compute_direction(self, strategy_a: str, strategy_b: str) -> Optional[np.ndarray]:
        """Compute average direction vector from strategy A to strategy B.
        
        Args:
            strategy_a: Name of the first strategy
            strategy_b: Name of the second strategy
        """
        pass