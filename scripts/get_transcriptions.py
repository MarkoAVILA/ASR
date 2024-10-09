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

def get_tr(file_channels, map, dir):
    path_audio, _ = read_files(file_channels)
    AUDIOS, TR  = read_files(map)
    path_audio_ = [i.split('/')[1].split('.')[0]+'_'+i.split('/')[1].split('.')[1].split('.wav')[0] for i in path_audio]
    output_audio = [dir+i.split('/')[1].split('.wav')[0]+'.tr' for i in path_audio]
    for idx, i in enumerate(path_audio_):
        fout = open(output_audio[idx], 'w+')
        for j,z in zip(AUDIOS,TR):
            if i in j:
                fout.write(z.strip()+'\n')

if __name__ == '__main__':
    fire.Fire(get_tr)
