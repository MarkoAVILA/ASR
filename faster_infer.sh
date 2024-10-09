# echo "Converting hf whisper to ct2 whisper ..."
# MODEL="/nfs/RESEARCH/avila/Projects/SPEECH2TEXT/ASR/ASR_PHONE/whisper-ft-small/checkpoint-24000" #"/nfs/RESEARCH/avila/Projects/SPEECH2TEXT/ASR/ASR_PHONE/whisper-ft-medium_2/checkpoint-28000"
MODEL_CT2="whisper-ft-small-ct2" #"whisper-ft-medium-ct2"
# ct2-transformers-converter --model $MODEL --output_dir $MODEL_CT2 --quantization float16
# echo "Done!"
echo "Faster Inference with Whisper"
# CUDA_VISIBLE_DEVICES=1 python3 scripts/faster_infer_custom_dataset.py generate --path_data segments_channels/separate_audio.txt --model_name $MODEL_CT2 --patern 'medium-ft'
CUDA_VISIBLE_DEVICES=1 python3 scripts/faster_infer_custom_dataset.py generate --path_data segments_channels/separate_audio.txt --model_name $MODEL_CT2 --patern 'small-ft'