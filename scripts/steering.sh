python src/run_experiment.py \
        --steering \
        --config config/Llama70B.yaml \
        --steer_layers layers.50\
        --topk 10 \
        --k_index 0 \
        --device cuda:0 \
        --get_index_type Reason \
        --steering_target_strategy cot \
        --token_pos 1 \
        --steer_n_steps 1 \
        --steer_alpha 20.0

# python src/run_experiment.py \
#         --steering \
#         --config config/Qwen0.6B.yaml \
#         --steer_layers layers.19\
#         --topk 25 \
#         --device cuda:3 \
#         --get_index_type Reason \
#         --steering_target_strategy cot \
#         --token_pos 1 \
#         --steer_n_steps 1 \
        # --steer_alpha 100.0

# python src/run_experiment.py \
#         --steering \
#         --config config/Qwen4B.yaml \
#         --steer_layers layers.17\
#         --topk 25 \
#         --device cuda:1 \
#         --get_index_type Reason \
#         --steering_target_strategy cot \
#         --token_pos 1 \
#         --steer_n_steps 1 \
#         --steer_alpha 85.0


# python src/run_experiment.py \
#         --steering \
#         --config config/Gemma4B.yaml \
#         --steer_layers layers.29\
#         --topk 100 \
#         --device cuda:1 \
#         --get_index_type Reason \
#         --steering_target_strategy direct \
#         --token_pos 1 \
#         --steer_n_steps 1 \
#         --steer_alpha 15.0

# python src/run_experiment.py \
#         --steering \
#         --config config/Gemma27B.yaml \
#         --steer_layers layers.53\
#         --topk 100 \
#         --device cuda:1 \
#         --get_index_type Reason \
#         --steering_target_strategy direct \
#         --token_pos 0 \
#         --steer_n_steps 2 \
#         --steer_alpha 10.0