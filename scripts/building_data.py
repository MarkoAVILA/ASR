import torchaudio
import tqdm
import numpy as np
from datasets import Dataset

def map_to_array(path):
    speech_array, sr = torchaudio.load(path)
    x = speech_array.numpy()
    return x, sr

def open_file(file):
    l_path, l_audio, l_transcription, l_sr = [], [], [], []
    with open(file, 'r') as f:
        for i in tqdm.tqdm(f):
            x = i.strip().split('\t')
            print(x)
            audio, sr = map_to_array(x[0].strip())
            print(audio)
            print(sr)
            l_path.append(x[0])
            l_audio.append(np.array(audio[0]))
            l_sr.append(sr)
            l_transcription.append(x[2])
    return l_path, l_audio, l_transcription, l_sr

def main(data_file="segments_filtered.map",
         output_train='data/train.hf',
         output_val='data/val.hf'):
    print('loading...')
    a, b,c,d = open_file(data_file)
    print('done1')
    ds = Dataset.from_dict({"audio": [{"path":str(x), "array":np.array(y), "sampling_rate":z} for (x,y),z in zip(zip(a,b),d)], "transcription": c, "tgt_lang":["french"]*len(c)})
    ds = ds.shuffle(seed=42)
    print('shuffled!')
    ds_test = ds.select(range(0,1000))
    ds_train = ds.select(range(1000,len(ds)))
    ds_train.save_to_disk(output_train)
    ds_test.save_to_disk(output_val)
    print('saved!')

def test(data_file, output):
    a, b,c,d = open_file(data_file)
    ds = Dataset.from_dict({"audio": [{"path":str(x), "array":np.array(y), "sampling_rate":z} for (x,y),z in zip(zip(a,b),d)], "transcription": c, "tgt_lang":["french"]*len(c)})
    ds.save_to_disk(output)


if __name__=='__main__':
    main(data_file="segments.map",
         output_train='new_data/train.hf',
         output_val='new_data/val.hf')
    
    # test("segments.map", 'test.hf')


