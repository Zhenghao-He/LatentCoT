#!/bin/bash
# SAE分析 (需要先有激活值文件)

python src/run_experiment.py \
        --run_analysis \
        --config config/default.yaml \
        --hook_layers layers.1 layers.5 layers.15 layers.19 layers.21 layers.24 layers.29 \
        --topk 100 \
        --device cuda:0 \
        --get_index_type Reason \
        --token_pos 2