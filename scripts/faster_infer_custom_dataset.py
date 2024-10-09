from faster_whisper import WhisperModel
from rich import print
import fire

def read_files(path_file):
    l_audio, l_tr = [],[]
    with open(path_file,'r') as f:
        for i in f:
            x = i.split('\t')
            if len(x)>1:
                l_audio.append(x[0])
                l_tr.append(x[1])
            else:
                l_audio.append(x[0])
    return l_audio, l_tr


class WHISPERTRANCRIPT:
    def __init__(self, path_data, model_name="large-v3", device="cuda", compute_type="float16", timestamps=False, patern='medium') -> None:
        self.audio_path, self.transcriptions = read_files(path_data)
        self.output_path = [i.split('.wav')[0]+f".{patern}"+'.tr' for i in self.audio_path]
        # Run on GPU with FP16
        self.device = device
        self.compute_type = compute_type
        self.timestamps = timestamps
        self.model = WhisperModel(model_name, device=self.device, compute_type=self.compute_type)
    
    def generate(self):
        for idx, ap in enumerate(self.audio_path):
            f_out = open(self.output_path[idx],'w')
            segments, info = self.model.transcribe(ap.strip(), beam_size=5, vad_filter=True)
            print("Detected language %s with probability %f", info.language, info.language_probability)
            transcripts = []
            for i in segments:
                start_, end_, tr = i.start,i.end, i.text
                if self.timestamps:
                    transcripts.append("[%.2fs -> %.2fs] %s" % (start_,end_,tr))
                    f_out.write("[%.2fs -> %.2fs] %s" % (start_,end_,tr) + '\n')
                else:
                    transcripts.append(tr.strip())
                    f_out.write(tr.strip()+'\n')
            tr = "\n".join(transcripts)
            print(tr)

if __name__=='__main__':
    fire.Fire(WHISPERTRANCRIPT)