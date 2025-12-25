#!/bin/bash
# 提取激活值

# cd src
# python src/run_experiment.py \
#         --extract_latent_zs \
#         --config config/Qwen0.6B.yaml \
#         --hook_layers layers.19 \
#         --device cuda:2 \
#         --token_pos 1 \

# python src/run_experiment.py \
#         --extract_latent_zs \
#         --config config/Qwen4B.yaml \
#         --hook_layers layers.21 \
#         --device cuda:2 \
#         --token_pos 1 \

python src/run_experiment.py \
        --extract_latent_zs \
        --config config/Gemma4B.yaml \
        --hook_layers layers.22 \
        --device cuda:0 \
        --token_pos 1 \