#!/bin/bash
# 提取激活值

# cd src
# python src/run_experiment.py \
#         --run_baseline \
#         --config config/Llama70B.yaml \
#         --multi_gpu \
#         --device cuda:5 \
#         --token_pos 1 \

# python src/run_experiment.py \
#         --run_baseline \
#         --config config/Llama8B.yaml \
#         --device cuda:2 \
#         --token_pos 1 \

python src/run_experiment.py \
        --run_baseline \
        --config config/Llama8B_R1_distill.yaml \
        --device cuda:1 \
        --token_pos 1 \

# python src/run_experiment.py \
#         --run_baseline \
#         --config config/Qwen0.6B.yaml \
#         --device cuda:5 \
#         --token_pos 1 \

# python src/run_experiment.py \
#         --run_baseline \
#         --config config/Qwen4B.yaml \
#         --device cuda:2 \
#         --token_pos 1 \

# python src/run_experiment.py \
#         --run_baseline \
#         --config config/Gemma4B.yaml \
#         --device cuda:1 \
#         --token_pos 1 \

# python src/run_experiment.py \
#         --run_baseline \
#         --multi_gpu \
#         --config config/Gemma12B.yaml \
#         --device cuda:6 \
#         --token_pos 1 \


# python src/run_experiment.py \
#         --extract_latent_zs \
#         --config config/Gemma27B.yaml \
#         --device cuda:2 \
#         --token_pos 0 \

