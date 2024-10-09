echo "Get True Transcriptions ..."
FCH="/nfs/RESEARCH/avila/Projects/SPEECH2TEXT/ASR/ASR/segments_channels/separate_audio.txt"
MAP="/nfs/RESEARCH/avila/Projects/SPEECH2TEXT/ASR/ASR/data/new_test.map"
DIR="segments_channels/"
python3 scripts/get_transcriptions.py --file_channels $FCH --map $MAP --dir $DIR
echo "Done!"