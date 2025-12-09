"""Utilities for setting random seeds and ensuring reproducibility."""

import torch
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)


def set_random_seed(seed: int = 42, verbose: bool = True) -> None:
    """Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        verbose: Whether to log the seed setting
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if verbose:
        logger.info(f"Random seed set to {seed} for reproducibility")


def get_random_seed_from_config(config, default: int = 42) -> int:
    """Get random seed from config with fallback to default.
    
    Args:
        config: Configuration object with .get() method
        default: Default seed value if not found in config
        
    Returns:
        Random seed value
    """
    if hasattr(config, 'get'):
        return config.get('experiment.random_seed', default)
    elif isinstance(config, dict):
        return config.get('experiment', {}).get('random_seed', default)
    else:
        return default


def setup_reproducibility(config=None, seed: int = None) -> int:
    """Setup reproducibility with automatic seed detection from config.
    
    Args:
        config: Configuration object (optional)
        seed: Manual seed override (optional)
        
    Returns:
        The seed value that was set
    """
    if seed is None:
        seed = get_random_seed_from_config(config) if config else 42
    
    set_random_seed(seed)
    return seed
