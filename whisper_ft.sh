echo "Training Architecture Whisper without adapters"
OUTPUT_DIR="./whisper-ft-base" #"./whisper-ft-medium" #"./whisper-fr-large"
WHISPER_MODEL="openai/whisper-base"  # openai/whisper-medium #"openai/whisper-large-v2" #"openai/whisper-small"
LAN="french"
DATA="/nfs/RESEARCH/avila/Projects/SPEECH2TEXT/ASR/ASR_PHONE/data"
CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/whisper_ft_custom_dataset.py train --path_model $WHISPER_MODEL --data_train $DATA/segments_trn.map --data_val $DATA/segments_val.map --num_proc 20 --langue_1 $LAN --output_dir $OUTPUT_DIR --per_device_train_batch_size 32  --per_device_eval_batch_size 32 --gradient_accumulation_steps 1 --learning_rate 2.5e-5 --max_steps 100000000 --save_steps 1000 --eval_steps 1000 --push_to_hub False > logs/log_ft_base &
