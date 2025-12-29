#!/bin/bash
# 提取激活值

# cd src
# python src/run_experiment.py \
#         --extract_latent_zs \
#         --config config/Qwen0.6B.yaml \
#         --hook_layers layers.19 \
#         --device cuda:5 \
#         --token_pos 1 \

# python src/run_experiment.py \
#         --extract_latent_zs \
#         --config config/Qwen4B.yaml \
#         --hook_layers layers.17 layers.19 layers.23 layers.25 layers.27 layers.29 layers.31 layers.33  \
#         --device cuda:4 \
#         --token_pos 1 \

# python src/run_experiment.py \
#         --extract_latent_zs \
#         --config config/Gemma4B.yaml \
#         --hook_layers layers.9 layers.17 layers.22 layers.29\
#         --device cuda:1 \
#         --token_pos 1 \

python src/run_experiment.py \
        --extract_latent_zs \
        --config config/Gemma12B.yaml \
        --hook_layers layers.41\
        --device cuda:6 \
        --token_pos 1 \


# python src/run_experiment.py \
#         --extract_latent_zs \
#         --config config/Gemma27B.yaml \
#         --hook_layers layers.31 layers.40 layers.53 \
#         --device cuda:2 \
#         --token_pos 0 \

