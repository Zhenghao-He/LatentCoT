# python src/run_experiment.py \
#         --steering \
#         --config config/Llama70B.yaml \
#         --steer_layers layers.50\
#         --topk 10 \
#         --k_index 0 \
#         --device cuda:0 \
#         --multi_gpu \
#         --get_index_type Reason \
#         --steering_target_strategy cot \
#         --token_pos 1 \
#         --steer_n_steps 1 \
#         --steer_alpha 20

python src/run_experiment.py \
        --steering \
        --config config/Llama8B.yaml \
        --steer_layers layers.19\
        --topk 1000 \
        --k_index 3 \
        --device cuda:5 \
        --get_index_type Reason \
        --steering_target_strategy direct \
        --token_pos 1 \
        --steer_n_steps 6 \
        --steer_alpha 15

# python src/run_experiment.py \
#         --steering \
#         --config config/Qwen0.6B.yaml \
#         --steer_layers layers.19\
#         --topk 25 \
#         --device cuda:1 \
#         --get_index_type Reason \
#         --steering_target_strategy cot \
#         --token_pos 1 \
#         --steer_n_steps -1 \
#         --steer_alpha 25.0

# python src/run_experiment.py \
#         --steering \
#         --config config/Qwen4B.yaml \
#         --steer_layers layers.29\
#         --topk 25 \
#         --device cuda:5 \
#         --get_index_type Reason \
#         --steering_target_strategy direct \
#         --token_pos 1 \
#         --steer_n_steps 10 \
#         --steer_alpha 50.0




# python src/run_experiment.py \
#         --steering \
#         --config config/Gemma4B.yaml \
#         --steer_layers layers.29\
#         --topk 100 \
#         --device cuda:4 \
#         --get_index_type Reason \
#         --steering_target_strategy direct \
#         --token_pos 1 \
#         --steer_n_steps 1 \
#         --steer_alpha 15.0



# python src/run_experiment.py \
#         --steering \
#         --config config/Gemma12B.yaml \
#         --steer_layers layers.31\
#         --topk 100 \
#         --multi_gpu \
#         --device cuda:6 \
#         --get_index_type Reason \
#         --steering_target_strategy cot \
#         --token_pos 1 \
#         --steer_n_steps 1 \
#         --steer_alpha 25.0

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