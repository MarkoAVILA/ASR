import torch
import torchaudio as torchaud_
import fire

def separate_in_channels(audio_path, dir):
    waveform, sr = torchaud_.load(audio_path)
    # resampler = torchaud_.transforms.Resample(orig_freq=sr, new_freq=16000)
    channel_0 = waveform[0, :]
    channel_1 = waveform[1, :]
    x = audio_path.split('/')[-1]
    base_path = x.split('.wav')[0].strip()
    torchaud_.save(dir+base_path+f'.ch{0}'+'.wav', channel_0.unsqueeze(0), sr)
    torchaud_.save(dir+base_path+f'.ch{1}'+'.wav', channel_1.unsqueeze(0), sr)
    return dir+base_path+f'.ch{0}'+'.wav', dir+base_path+f'.ch{1}'+'.wav'


def main(path_list, dir):
    with open(path_list,'r') as f, open(dir+'separate_audio.txt','w+') as f1:
        for i in f:
            ch0,ch1 = separate_in_channels(i.strip(), dir)
            f1.write(str(ch0)+'\n')
            f1.write(str(ch1)+'\n')



if __name__=='__main__':
    fire.Fire(main)
