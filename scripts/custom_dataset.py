from datasets import Dataset, IterableDataset
import tqdm
from rich import print
import fire 
from datasets import Audio, Value

MAX_INPUT_LENGTH = 30.0

def read_file_segments(file):
    l_path_audio, l_transcription = [], []
    with open(file,'r') as f:
        for i in tqdm.tqdm(f):
            x = i.strip().split('\t')
            l_path_audio.append(x[0])
            l_transcription.append(str(x[1]).strip())
    return l_path_audio, l_transcription

def building_dataset(file, output):
    print('Loading....')
    audio_path,transcriptions = read_file_segments(file)
    lang = ['french']
    print('Building...')
    ds = Dataset.from_dict({"audio": audio_path, 'transcription': transcriptions, 'tgt_lang':lang*len(transcriptions)})
    print('Casting...')
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.cast_column('transcription', Value('string'))
    ds.save_to_disk(output)
    print('Saved!')

def get_dataset(file):
    print('Loading....')
    audio_path,transcriptions = read_file_segments(file)
    lang = ['french']
    print('Building...')
    ds = Dataset.from_dict({"audio": audio_path, 'transcription': transcriptions, 'tgt_lang':lang*len(transcriptions)})
    print('Casting...')
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.cast_column('transcription', Value('string'))
    return ds

def is_audio_in_length_range(example):
    audio, sr = example['audio']['array'], example['audio']['sampling_rate']
    return len(audio) / sr < MAX_INPUT_LENGTH


def iter_dataset(file_map, type='train'):
    print('Loading....')
    audio_path,transcriptions = read_file_segments(file_map)
    lang = ['french']
    print('Building...')
    ds = Dataset.from_dict({"audio": audio_path, 'transcription': transcriptions, 'tgt_lang':lang*len(transcriptions)})
    if type=='train':
        ds = ds.shuffle(seed=42)
        print('dataset shuffled!', flush=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.cast_column('transcription', Value('string'))
    ds = ds.cast_column('tgt_lang', Value('string'))
    ds = ds.to_iterable_dataset()
    if type=='train':
        ds = ds.filter(is_audio_in_length_range)
    print('Iterable dataset done!')
    return ds


if __name__=='__main__':
    fire.Fire(building_dataset)


