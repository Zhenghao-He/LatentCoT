#!/bin/bash
# 提取激活值

# cd src
python src/run_experiment.py \
        --extract_latent_zs \
        --config config/large.yaml \
        --hook_layers layers.50 \
        --device cuda:0 \
        --token_pos 1 \