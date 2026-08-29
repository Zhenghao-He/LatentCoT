"""Main entry point for running planning analysis experiments."""

import argparse

from runner import ExperimentRunner
from my_datasets.dataset_loader import DataLoader
from my_datasets.GSM8KLoader import GSM8KLoader
from my_datasets.GPQALoader import GPQALoader
from my_datasets.MMLULoader import MMLULoader
from my_datasets.BBHLoader import BBHLoader
from my_datasets.MATHLoader import MATHLoader
from my_datasets.MATH500Loader import MATH500Loader
# run_experiment.py 顶部，第一屏代码
import torch
# torch._dynamo.disable()
# torch._dynamo.config.cache_size_limit = 32
from utils.config import Config
from analysis.LatentAnalyzer import LatentAnalyzer

def main():
    """Main function for running experiments."""
    # import torch._dynamo as dynamo
    # print("dynamo config enabled:", dynamo.config.enabled)
    parser = argparse.ArgumentParser(description="Run planning analysis experiments")
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--multi_gpu',
        action='store_true',
        help='Flag to enable multi-GPU training'
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
    parser.add_argument(
        '--k_index',
        type=int,
        default=None,
        help='index K for top-K latent representation extraction'
    )
    parser.add_argument(
        '--steer_layers',
        type=str,
        nargs='*',  # 允许0个或多个参数
        default=None,
        help='List of layers to hook for activation extraction (e.g., --steer_layers layers.24 layers.25)'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=20,
        help='Top K indices to consider for analysis'
    )
    parser.add_argument(
        '--get_index_type',
        type=str,
        default='Reason',
        help='Type of index to get (e.g., Reason or Hint)'
    )
    parser.add_argument(
        '--steering_target_strategy',
        type=str,
        choices=['direct', 'cot', 'hint', 'cot_hint'],
        help='Target strategy for steering experiment'
    )
    parser.add_argument(
        '--steer_alpha',
        type=float,
        default=1.0,
        help='Steering strength alpha value'
    )
    parser.add_argument(
        '--steer_n_steps',
        type=int,
        default=1,
        help='Number of steps to apply steering'
    )
    parser.add_argument(
        '--max_activation_length',
        type=int,
        default=None,
        help='Maximum length of activations to consider'
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
        '--dense_steering',
        action='store_true',
        help='Flag to run steering experiment'
    )
    run_group.add_argument(
        '--run_experiment',
        action='store_true',
        help='Flag to run the full experiment'
    )
    run_group.add_argument(
        '--run_baseline',
        action='store_true',
        help='Flag to run baseline experiments'
    )
    run_group.add_argument(
        '--record_activations',
        action='store_true',
        help='Flag to record model activations'
    )
    run_group.add_argument(
        '--eval_features',
        action='store_true',
        help='Flag to evaluate features'
    )
    run_group.add_argument(
        '--construct_dense_direction',
        action='store_true',
        help='Flag to extract raw activations without any pooling or analysis'
    )
    run_group.add_argument(
        '--get_acts_for_each_sample',
        action='store_true',
        help='Flag to get activations for each sample individually'
    ) # max acts per sample

    args = parser.parse_args()
    
    config = Config(args.config)
    
    if args.extract_latent_zs:
        dt_name = config.get('dataset.name', '')
        if dt_name == 'gsm8k':
            dt_loader = GSM8KLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        elif dt_name == 'gpqa':
            dt_loader = GPQALoader(
                base_path=config.get('dataset.paths', ''),
                data_subset=config.get('dataset.data_subset', 'gpqa_diamond'),
                max_samples=config.get('dataset.max_samples', None)
            )
        elif dt_name == 'mmlu':
            dt_loader = MMLULoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None),
                data_subset=config.get('dataset.data_subset', 'all')
            )
        elif dt_name == 'bbh':
            dt_loader = BBHLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        else:
            dt_loader = DataLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        runner = ExperimentRunner(args.config, args=args, data_loader=dt_loader)
        runner.extract_latent_zs()
        print("Latent extraction completed")
    elif args.run_analysis:
        
        analyzer = LatentAnalyzer(config, args)
        analyzer.load_latents()
        
        # Example: compute direction between 'direct' and 'cot'
        analyzer.compute_direction_by_difference(args.get_index_type)
        print("Latent analysis completed")

    elif args.steering:
        dt_name = config.get('dataset.name', '')
        if dt_name == 'gsm8k':
            dt_loader = GSM8KLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        elif dt_name == 'gpqa':
            dt_loader = GPQALoader(
                base_path=config.get('dataset.paths', ''),
                data_subset=config.get('dataset.data_subset', 'gpqa_diamond'),
                max_samples=config.get('dataset.max_samples', None)
            )
        elif dt_name == 'mmlu':
            dt_loader = MMLULoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None),
                data_subset=config.get('dataset.data_subset', 'all')
            )
        elif dt_name == 'bbh':
            dt_loader = BBHLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        else:
            dt_loader = DataLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )

        runner = ExperimentRunner(args.config, args=args, data_loader=dt_loader)
        runner.run_steering_experiment()
        print("Steering experiment completed")

    elif args.dense_steering:
        dt_name = config.get('dataset.name', '')
        if dt_name == 'gsm8k':
            dt_loader = GSM8KLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        elif dt_name == 'gpqa':
            dt_loader = GPQALoader(
                base_path=config.get('dataset.paths', ''),
                data_subset=config.get('dataset.data_subset', 'gpqa_diamond'),
                max_samples=config.get('dataset.max_samples', None)
            )
        elif dt_name == 'mmlu':
            dt_loader = MMLULoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None),
                data_subset=config.get('dataset.data_subset', 'all')
            )
        elif dt_name == 'bbh':
            dt_loader = BBHLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        else:
            dt_loader = DataLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )

        runner = ExperimentRunner(args.config, args=args, data_loader=dt_loader)
        runner.run_dense_steering_experiment()
        print("Steering experiment completed")
    elif args.run_baseline:
        dt_name = config.get('dataset.name', '')
        if dt_name == 'gsm8k':
            dt_loader = GSM8KLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        elif dt_name == 'gpqa':
            dt_loader = GPQALoader(
                base_path=config.get('dataset.paths', ''),
                data_subset=config.get('dataset.data_subset', 'gpqa_diamond'),
                max_samples=config.get('dataset.max_samples', None)
            )
        elif dt_name == 'mmlu':
            dt_loader = MMLULoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None),
                data_subset=config.get('dataset.data_subset', 'all')
            )
        elif dt_name == 'bbh':
            dt_loader = BBHLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )    
        elif dt_name == 'math':
            dt_loader = MATHLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None),
                data_subset=config.get('dataset.data_subset', 'math'),
                balance_by_level=config.get('dataset.balance_by_level', False),
                samples_per_level=config.get('dataset.samples_per_level', None),
                fill_shortfall_to_max=config.get('dataset.fill_shortfall_to_max', False),
                random_seed=config.get('experiment.random_seed', 42),
                allowed_levels=config.get('dataset.allowed_levels', None),
            )
        elif dt_name == 'math500':
            dt_loader = MATH500Loader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None),
                data_subset=config.get('dataset.data_subset', 'math500'),
                balance_by_level=config.get('dataset.balance_by_level', False),
                samples_per_level=config.get('dataset.samples_per_level', None),
                fill_shortfall_to_max=config.get('dataset.fill_shortfall_to_max', False),
                random_seed=config.get('experiment.random_seed', 42),
                allowed_levels=config.get('dataset.allowed_levels', None),
            )
        else:
            dt_loader = DataLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        runner = ExperimentRunner(args.config, args=args, data_loader=dt_loader)
        runner.run_baseline()
        print("Baseline experiment completed")
    elif args.record_activations:
        dt_name = config.get('dataset.name', '')
        if dt_name == 'gsm8k':
            dt_loader = GSM8KLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        elif dt_name == 'gpqa':
            dt_loader = GPQALoader(
                base_path=config.get('dataset.paths', ''),
                data_subset=config.get('dataset.data_subset', 'gpqa_diamond'),
                max_samples=config.get('dataset.max_samples', None)
            )
        elif dt_name == 'mmlu':
            dt_loader = MMLULoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None),
                data_subset=config.get('dataset.data_subset', 'all')
            )
        elif dt_name == 'bbh':
            dt_loader = BBHLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        else:
            dt_loader = DataLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        runner = ExperimentRunner(args.config, args=args, data_loader=dt_loader)
        runner.record_activations()
        print("Activation recording completed")
    elif args.eval_features:
        dt_name = config.get('dataset.name', '')
        if dt_name == 'gsm8k':
            dt_loader = GSM8KLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        elif dt_name == 'gpqa':
            dt_loader = GPQALoader(
                base_path=config.get('dataset.paths', ''),
                data_subset=config.get('dataset.data_subset', 'gpqa_diamond'),
                max_samples=config.get('dataset.max_samples', None)
            )
        elif dt_name == 'mmlu':
            dt_loader = MMLULoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None),
                data_subset=config.get('dataset.data_subset', 'all')
            )
        elif dt_name == 'bbh':
            dt_loader = BBHLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        else:
            dt_loader = DataLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        runner = ExperimentRunner(args.config, args=args, data_loader=dt_loader)
        runner.eval_latent_activations()
        print("Feature evaluation completed")
    elif args.get_acts_for_each_sample:
        dt_name = config.get('dataset.name', '')
        if dt_name == 'gsm8k':
            dt_loader = GSM8KLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        elif dt_name == 'gpqa':
            dt_loader = GPQALoader(
                base_path=config.get('dataset.paths', ''),
                data_subset=config.get('dataset.data_subset', 'gpqa_diamond'),
                max_samples=config.get('dataset.max_samples', None)
            )
        elif dt_name == 'mmlu':
            dt_loader = MMLULoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None),
                data_subset=config.get('dataset.data_subset', 'all')
            )
        elif dt_name == 'bbh':
            dt_loader = BBHLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        else:
            dt_loader = DataLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        runner = ExperimentRunner(args.config, args=args, data_loader=dt_loader)
        runner.record_all_activations()
        print("Get activations for each sample completed")

    elif args.construct_dense_direction:
        dt_name = config.get('dataset.name', '')
        if dt_name == 'gsm8k':
            dt_loader = GSM8KLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        elif dt_name == 'gpqa':
            dt_loader = GPQALoader(
                base_path=config.get('dataset.paths', ''),
                data_subset=config.get('dataset.data_subset', 'gpqa_diamond'),
                max_samples=config.get('dataset.max_samples', None)
            )
        elif dt_name == 'mmlu':
            dt_loader = MMLULoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None),
                data_subset=config.get('dataset.data_subset', 'all')
            )
        elif dt_name == 'bbh':
            dt_loader = BBHLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        else:
            dt_loader = DataLoader(
                base_path=config.get('dataset.paths', ''),
                max_samples=config.get('dataset.max_samples', None)
            )
        runner = ExperimentRunner(args.config, args=args, data_loader=dt_loader)
        runner.construct_dense_direction()
        print("Get raw activations for each sample completed")
    else:
        raise ValueError("No valid run option selected.")


    





if __name__ == "__main__":
    main()