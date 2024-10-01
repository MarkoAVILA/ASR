echo "Training Architecture Whisper without adapters"
OUTPUT_DIR="./whisper-ft-medium" #"./whisper-fr-large"
WHISPER_MODEL="openai/whisper-medium" #"openai/whisper-large-v2" #"openai/whisper-small"
LAN="french" 
nohup python3 scripts/whisper_ft.py train --path_model $WHISPER_MODEL --data_train data/train.hf --data_val data/val.hf --num_proc 20 --langue_1 $LAN --output_dir $OUTPUT_DIR --per_device_train_batch_size 8  --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 --max_steps 100000000 --save_steps 500 --eval_steps 100 --push_to_hub False > logs/log_ft_medium &
