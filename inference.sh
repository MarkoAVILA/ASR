echo 'Inference with Whisper Models'
MAP_DATA='data/test.map'
MODEL_NAME='openai/whisper-medium'
TASK='transcribe'
LAN='french'
OUT='transcriptions/pckt.whisper.medium'
nohup python3 scripts/infer_custom_dataset.py inference --map_data $MAP_DATA --model_name $MODEL_NAME --task $TASK --language $LAN --output $OUT > logs/pckt_infer.whisper.medium & 