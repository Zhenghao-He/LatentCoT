#!/bin/bash
# SAE分析 (需要先有激活值文件)

python src/run_experiment.py \
        --run_analysis \
        --config config/Llama8B.yaml \
        --hook_layers layers.19 \
        --topk 25 \
        --device cuda:0 \
        --get_index_type Reason \
        --token_pos 1

# python src/run_experiment.py \
#         --run_analysis \
#         --config config/Qwen4B.yaml \
#         --hook_layers layers.17 layers.19 layers.23 layers.25 layers.27 layers.29 layers.31 layers.33 \
#         --topk 2000 \
#         --device cuda:1 \
#         --get_index_type Reason \
#         --token_pos 1

# python src/run_experiment.py \
#         --run_analysis \
#         --config config/Gemma4B.yaml \
#         --hook_layers layers.9 layers.17 layers.22 layers.29 \
#         --topk 100 \
#         --device cuda:1 \
#         --get_index_type Reason \
#         --token_pos 1

# python src/run_experiment.py \
#         --run_analysis \
#         --config config/Gemma12B.yaml \
#         --hook_layers layers.12 layers.24 layers.31 layers.41 \
#         --topk 100 \
#         --device cuda:1 \
#         --get_index_type Reason \
#         --token_pos 1

# python src/run_experiment.py \
#         --run_analysis \
#         --config config/Gemma27B.yaml \
#         --hook_layers layers.31 layers.40 layers.53 \
#         --topk 100 \
#         --device cuda:1 \
#         --get_index_type Reason \
#         --token_pos 0
