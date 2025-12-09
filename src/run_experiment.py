"""Main entry point for running planning analysis experiments."""

import argparse

from runner import ExperimentRunner
from my_datasets.dataset_loader import DataLoader, GSM8KLoader
import torch
from utils.config import Config

def main():
    """Main function for running experiments."""
    parser = argparse.ArgumentParser(description="Run planning analysis experiments")
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run the experiment on (e.g., "cuda" or "cpu")'
    )
    parser.add_argument(
        '--hook_layers',
        type=str,
        nargs='*',  # 允许0个或多个参数
        default=None,
        help='List of layers to hook for activation extraction (e.g., --hook_layers layers.24 layers.25)'
    )

    analysis_group = parser.add_mutually_exclusive_group(required=True)
    analysis_group.add_argument(
        '--token_pos',
        type=int,
        help='Token position to extract hidden states from (e.g., -1 for last token)'
    )
    analysis_group.add_argument(
        '--type_of_analysis',
        type=str,
        choices=['avg_pooling', 'max_pooling'],
        help='Type of analysis to perform'
    )
    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument(
        '--extract_latent_zs',
        action='store_true',
        help='Flag to extract latent representations'
    )
    run_group.add_argument(
        '--run_analysis',
        action='store_true',
        help='Flag to run analysis on extracted representations'
    )
    run_group.add_argument(
        '--steering',
        action='store_true',
        help='Flag to run steering experiment'
    )
    run_group.add_argument(
        '--run_experiment',
        action='store_true',
        help='Flag to run the full experiment'
    )
    args = parser.parse_args()
    
    config = Config(args.config)
    dt_name = config.get('dataset.name', '')
    if dt_name == 'gsm8k':
        dt_loader = GSM8KLoader(
            base_path=config.get('dataset.paths', ''),
            max_samples=config.get('dataset.max_samples', None)
        )
    else:
        dt_loader = DataLoader(
            base_path=config.get('dataset.paths', ''),
            max_samples=config.get('dataset.max_samples', None)
        )

    runner = ExperimentRunner(args.config, args=args, data_loader=dt_loader)

    if args.extract_latent_zs:
        runner.extract_latent_zs()
        print("Latent extraction completed")
    elif args.run_analysis:
        runner.run_analysis()
        print("Latent analysis completed")

        


    





if __name__ == "__main__":
    main()