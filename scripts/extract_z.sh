#!/bin/bash
# 提取激活值

# cd src
python src/run_experiment.py \
        --extract_latent_zs \
        --config config/default.yaml \
        --hook_layers layers.1 layers.5 layers.15 layers.19 layers.21 layers.24 layers.29 \
        --device cuda:0 \
        --token_pos -1