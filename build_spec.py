import os
import numpy as np
import random
import pickle
import math
from tqdm import tqdm

import torch
import torchaudio
import torchaudio.transforms as transforms
from torch.utils.data.dataset import Dataset

HOP_LENGTH = 400
SAMPLE_RATE = 16000

_SAMPLE_DIR = '/home/lijingru/bishe/IR/'
SAMPLE_RIR_PATH = os.path.join(_SAMPLE_DIR, "rir.wav")
WAVE_DIR = '/home/lijingru/bishe/fma_medium_wav'

def get_sample(path, resample=None):
    effects = [
    ["remix", "1"]
    ]
    if resample:
        effects.append(["rate", f'{resample}'])
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

def get_rir_sample(*, resample=None, processed=False):
    rir_raw, sample_rate = get_sample(SAMPLE_RIR_PATH, resample=resample)
    if not processed:
        return rir_raw, sample_rate
    rir = rir_raw[:, int(sample_rate*1.01):int(sample_rate*1.3)]
    rir = rir / torch.norm(rir, p=2)
    rir = torch.flip(rir, [1])
    return rir, sample_rate

def get_spec(sample_rate, waveform):
    mel_spectrogram = transforms.MelSpectrogram(sample_rate=sample_rate,
                                                n_fft=800,
                                                win_length=400,
                                                hop_length=HOP_LENGTH,
                                                power=2.0,
                                                f_min=55,
                                                f_max=7040,
                                                n_mels=84,
                                                )              
    mel_specgram = mel_spectrogram(waveform)
    return mel_specgram

def read_txt(path):
    lis = []
    with open(path, "r") as f:
        for line in f.readlines():                          #依次读取每行  
            line = line.strip()  
            lis.append(line)
    return lis

def readPkl(pklPath):
    with open(pklPath, "rb") as fp:
        lis = pickle.load(fp)   
    return lis

def savePkl(lis, pklPath):
    with open(pklPath, "wb") as fp:
        pickle.dump(lis, fp)

def ensureFolderExists(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)

def build_mel_spec(dump_mel_pth, filelist, root_path):
    speclist = []
    
    if os.path.exists(dump_mel_pth):
        print("pkl files already built")
        # speclist = readPkl(dump_mel_pth)
    else:
        print("pkl files not found, start building original mel specs ...")
        for paths in tqdm(filelist):
            wav_path = os.path.join(root_path, paths)
            waveform, sample_rate = get_sample(wav_path, resample=SAMPLE_RATE)
            mel_specgram = get_spec(sample_rate, waveform)
            speclist.append(mel_specgram)
        savePkl(speclist, dump_mel_pth)

def build_ir_spec(dump_mel_ir_pth, filelist, root_path, mode='train'):
    ir_speclist = []

    if os.path.exists(dump_mel_ir_pth):
        print("pkl files already built")
        # speclist = readPkl(dump_mel_ir_pth)
    else:
        print("pkl files not found, start building impulse response mel specs ...")
        save_dir = '/home/lijingru/bishe/test_wav/ir'
        ensureFolderExists(save_dir)
        rir_raw, _ = get_rir_sample(resample=SAMPLE_RATE)
        rir = rir_raw[:, int(SAMPLE_RATE*1.01):int(SAMPLE_RATE*1.3)]
        rir = rir / torch.norm(rir, p=2)
        rir = torch.flip(rir, [1])
        rir = rir.cuda()

        for paths in tqdm(filelist):
            wav_path = os.path.join(root_path, paths)
            waveform, sample_rate = get_sample(wav_path, resample=SAMPLE_RATE)
            # 这块理论上放到 GPU 更快？
            waveform = waveform.cuda()
            audio_ = torch.nn.functional.pad(waveform, (rir.shape[1]-1, 0))
            augmented = torch.nn.functional.conv1d(audio_[None, ...], rir[None, ...])[0]
            augmented = augmented.cpu()
            mel_specgram = get_spec(sample_rate, augmented)
            ir_speclist.append(mel_specgram)
            if mode == 'test':
                wav_name = paths.split('/')[-1]
                save_path = os.path.join(save_dir, wav_name)
                torchaudio.backend.sox_io_backend.save(save_path, augmented, SAMPLE_RATE)
        savePkl(ir_speclist, dump_mel_ir_pth)

def build_noise_spec(dump_mel_noise_pth, filelist, root_path, noise_path):
    noise_speclist = []
    infolist = []
    noise_lis = []

    if os.path.exists(dump_mel_noise_pth):
        print("pkl files already built")
        # noise_speclist = readPkl(dump_mel_noise_pth)
    else:
        print("pkl files not found, start building noisy mel specs ...")
        noiselist = os.listdir(noise_path)
        noise_path = os.path.join(noise_path, noiselist[0])
        noise, _ = get_sample(noise_path, resample=SAMPLE_RATE)

        for paths in tqdm(filelist):
            wav_path = os.path.join(root_path, paths)
            audio, sample_rate = get_sample(wav_path, resample=SAMPLE_RATE)
            # randomly choose a piece of noise
            noise_st = random.randint(0, noise.shape[1]-audio.shape[1]-1)
            noise_ed = noise_st + audio.shape[1]
            noise_cut = noise[:, noise_st:noise_ed]
            
            audio_power = audio.norm(p=2)
            noise_power = noise_cut.norm(p=2)

            cur_list = []
            for snr_db in [20, 10, 3]:
                snr = math.exp(snr_db / 10)
                noise_part = noise_cut * (audio_power/noise_power) * (1/(snr + 1))
                audio_part = audio * (1 - 1/(snr + 1))
                noisy_audio = noise_part + audio_part
                noisy_mel_spec = get_spec(sample_rate, noisy_audio)
                cur_list.append(noisy_mel_spec)

            noise_speclist.append(cur_list)
        
        savePkl(noise_speclist, dump_mel_noise_pth)

def build_data(dump_root, txt_path, noise_path, mode='test'):
    dump_mel_pth = os.path.join(dump_root, "mel_spec_original.pkl")
    dump_mel_ir_pth = os.path.join(dump_root, "mel_spec_ir.pkl")
    dump_mel_noise_pth = os.path.join(dump_root, "mel_spec_noise.pkl")
    filelist = read_txt(txt_path)
    build_mel_spec(dump_mel_pth, filelist, WAVE_DIR)
    build_ir_spec(dump_mel_ir_pth, filelist, WAVE_DIR, mode=mode)
    build_noise_spec(dump_mel_noise_pth, filelist, WAVE_DIR, noise_path)

if __name__ == "__main__":
    # 重构这一版干的事情是，把建谱和后续对谱的各种处理分拆出来
    train_pth = '/home/lijingru/bishe/DataSplit/train_lis.txt'
    test_pth = '/home/lijingru/bishe/DataSplit/test_lis.txt'
    train_root = '/home/lijingru/bishe/train_pkl'
    test_root = '/home/lijingru/bishe/test_pkl'
    train_noise = '/home/lijingru/bishe/noise'
    test_noise = '/home/lijingru/bishe/noise_test'
    build_data(train_root, train_pth, train_noise, mode='train')
    build_data(test_root, test_pth, test_noise, mode='test')
