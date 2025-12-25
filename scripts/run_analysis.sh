#!/bin/bash
# SAE分析 (需要先有激活值文件)

# python src/run_experiment.py \
#         --run_analysis \
#         --config config/default.yaml \
#         --hook_layers layers.50 \
#         --topk 25 \
#         --device cuda:0 \
#         --get_index_type Reason \
#         --token_pos 1

python src/run_experiment.py \
        --run_analysis \
        --config config/Qwen0.6B.yaml \
        --hook_layers layers.19 \
        --topk 25 \
        --device cuda:7 \
        --get_index_type Reason \
        --token_pos 1