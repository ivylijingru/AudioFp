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
from audiomentations import SpecFrequencyMask

HOP_LENGTH = 200
SAMPLE_RATE = 16000

def get_sample(path, resample=None, upsample=None):
    effects = [
    ["remix", "1"]
    ]
    if resample:
        effects.append(["rate", f'{resample}'])
    # 这一步：实际听一下音质差别
    if upsample:
        effects.append(["rate", f'{upsample}'])
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

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

def build_mel_spec(dump_mel_pth, filelist, mode='train'):
    # if the file already exists, load from path
    speclist = []
    infolist = []
    testlist = []

    if os.path.exists(dump_mel_pth):
        print("pkl files found, start loading original mel specs ...")
        speclist = readPkl(dump_mel_pth)
        print(len(speclist))

    if mode == 'test':
        for ii, mel_specgram in enumerate(tqdm(speclist)):
            # 这个是因为会传进来1/10长度的filelist
            if ii >= len(filelist):
                break
            wav_name = filelist[ii].split('/')[-1]
            song_name = wav_name.split('.')[0]

            cut_len = SAMPLE_RATE * 5 // HOP_LENGTH
            hop_len = SAMPLE_RATE * 1 // HOP_LENGTH
            for cut_start in range(0, mel_specgram.shape[2], hop_len):
                cut_end = cut_start + cut_len
                if cut_end >= mel_specgram.shape[2]:
                    break
                mel_out = mel_specgram[:,:,cut_start:cut_end]
                start_time = cut_start // hop_len
                testlist.append(mel_out)
                infolist.append((song_name, start_time))
        return testlist, infolist

    if mode == 'train':
        return speclist

def build_ir_spec(dump_mel_ir_pth, filelist, mode='train'):
    ir_speclist = []
    infolist = []
    ir_lis = []

    if os.path.exists(dump_mel_ir_pth):
        print("pkl files found, start loading noisy mel specs ...")
        ir_speclist = readPkl(dump_mel_ir_pth)
    
    # 这块应该移到 get item 里面
    if mode == 'recognize':
        for ii, mel_specgram in enumerate(tqdm(ir_speclist)):
            if ii >= len(filelist):
                break
            wav_name = filelist[ii].split('/')[-1]
            song_name = wav_name.split('.')[0]

            cut_len = SAMPLE_RATE * 5 // HOP_LENGTH
            hop_len = SAMPLE_RATE * 1 // HOP_LENGTH
            if mel_specgram.shape[2] <= cut_len:
                continue 
            start_time = random.randint(0, (mel_specgram.shape[2] - cut_len - 1) // hop_len)
            cut_start = start_time * hop_len
            cut_end = cut_start + cut_len
            ir_lis.append(mel_specgram[:,:,cut_start:cut_end])

            infolist.append((song_name, start_time))
        return ir_lis, infolist

    if mode == 'train':
        return ir_speclist

def build_noise_spec(dump_mel_noise_pth, filelist, mode='train', snr='20'):
    noise_speclist = []
    infolist = []
    noise_lis = []

    if os.path.exists(dump_mel_noise_pth):
        print("pkl files found, start loading noisy mel specs ...")
        noise_speclist = readPkl(dump_mel_noise_pth)
    
    # 这块应该移到 get item 里面
    if mode == 'recognize':
        for ii, pack in enumerate(tqdm(noise_speclist)):
            if ii >= len(filelist):
                break
            mel1, mel2, mel3 = pack
            wav_name = filelist[ii].split('/')[-1]
            song_name = wav_name.split('.')[0]

            cut_len = SAMPLE_RATE * 5 // HOP_LENGTH
            hop_len = SAMPLE_RATE * 1 // HOP_LENGTH
            if mel1.shape[2] <= cut_len:
                continue 
            start_time = random.randint(0, (mel1.shape[2] - cut_len - 1) // hop_len)
            cut_start = start_time * hop_len
            cut_end = cut_start + cut_len
            
            if snr == '20':
                noise_lis.append(mel1[:,:,cut_start:cut_end])
            if snr == '10':
                noise_lis.append(mel2[:,:,cut_start:cut_end])
            if snr == '3':
                noise_lis.append(mel3[:,:,cut_start:cut_end])
            infolist.append((song_name, start_time))

        return noise_lis, infolist

    if mode == 'train':
        return noise_speclist

class SpecData(Dataset):
    def __init__(self, mode, snr='20'):
        self.root_path = '/home/lijingru/bishe/fma_medium_wav'
        # self.filelist = os.listdir(self.root_path)
        self.mode = mode
        self.snr = snr
        if mode == "test" or mode == "recognize" or mode == "wave":
            txt_path = '/home/lijingru/bishe/DataSplit/test_lis.txt'
            dump_root = '/home/lijingru/bishe/test_pkl'

        if mode == "train":
            txt_path = '/home/lijingru/bishe/DataSplit/train_lis.txt'
            dump_root = '/home/lijingru/bishe/train_pkl'

        self.filelist = read_txt(txt_path)
        self.speclist = []
        self.irlist = []
        self.noise_speclist = []
        
        dump_mel_pth = os.path.join(dump_root, "mel_spec_before.pkl")
        dump_mel_ir_pth = os.path.join(dump_root, "mel_spec_ir.pkl")
        dump_mel_noise_pth = os.path.join(dump_root, "mel_spec_noise_before.pkl")

        if self.mode == "train":
            self.speclist = build_mel_spec(dump_mel_pth, self.filelist, mode=mode)
            # self.irlist = build_ir_spec(dump_mel_ir_pth, self.filelist, mode=mode)
            self.noise_speclist = build_noise_spec(dump_mel_noise_pth, self.filelist, mode=mode)
        elif self.mode == "test":
            self.testlist, self.infolist = build_mel_spec(dump_mel_pth, self.filelist[:len(self.filelist)//10], mode=mode)
        elif self.mode == "recognize":
            self.irlist = build_ir_spec(dump_mel_ir_pth, self.filelist[:len(self.filelist)//10], mode=mode)
            self.noise_speclist, self.infolist = build_noise_spec(dump_mel_noise_pth, 
                                                                  self.filelist[:len(self.filelist)//10], 
                                                                  mode=mode,
                                                                  snr = self.snr)
        mask_fraction = 0.05
        self.transform = SpecFrequencyMask(
            fill_mode="constant",
            fill_constant=0.0,
            min_mask_fraction=mask_fraction,
            max_mask_fraction=mask_fraction,
            p=1.0,
        )

    def __getitem__(self, item):
        '''Train mode, randomly choose a degraded sample'''
        if self.mode == "train":
            return self.get_mel_origin(item)
        
        '''Test mode'''
        if self.mode == "test":
            mel_specgram = self.testlist[item]
            song_info = self.infolist[item]
            return (mel_specgram, song_info)
        
        '''Recognize mode'''
        if self.mode == "recognize":
            mel_specgram = self.noise_speclist[item]
            song_info = self.infolist[item]
            return (mel_specgram, song_info)

    def __len__(self):
        if self.mode == "train":
            self.len = len(self.filelist)
        if self.mode == "test":
            self.len = len(self.testlist)
        if self.mode == "recognize":
            self.len = len(self.infolist)
        
        print(self.len)

        return self.len

    def get_mel_origin(self, item):
        mel_specgram = self.speclist[item]
        noisy_mel_specgram = self.noise_speclist[item]
        # ir_mel_specgram = self.irlist[item]

        # 随机按位置切
        cut_len = SAMPLE_RATE * 5 // HOP_LENGTH
        cut_start = random.randint(0, mel_specgram.shape[2]-cut_len-1)
        cut_end = cut_start + cut_len
        mel_out = mel_specgram[:,:,cut_start:cut_end]
        # use random option for choosing another view of the original version
        option = random.randint(0, 3)

        if option < 3:
            mel_corruput_out = noisy_mel_specgram[option][:,:,cut_start:cut_end]
        elif option == 3:
            mel_corruput_out = self.transform(mel_out[0].numpy())
            mel_corruput_out = np.expand_dims(mel_corruput_out, axis=0)
            mel_corruput_out = torch.from_numpy(mel_corruput_out)
        # elif option == 4:
        #     mel_corruput_out = ir_mel_specgram[:,:,cut_start:cut_end]

        return (mel_out, mel_corruput_out)

    def get_mel_combine(self, item):
        mel_specgram = self.speclist[item]
        noisy_mel_specgram = self.noise_speclist[item]
        ir_mel_specgram = self.irlist[item]

        # 随机按位置切
        cut_len = SAMPLE_RATE * 5 // HOP_LENGTH
        cut_start = random.randint(0, mel_specgram.shape[2]-cut_len-1)
        cut_end = cut_start + cut_len
        mel_out = mel_specgram[:,:,cut_start:cut_end]
        # use random option for choosing another view of the original version
        sample_lis = random.sample(range(0, 6), 2)
        ret_lis = []
        for option in sample_lis:
            if option < 3:
                mel_corruput_out = noisy_mel_specgram[option][:,:,cut_start:cut_end]
            elif option == 3:
                mel_corruput_out = self.transform(mel_out[0].numpy())
                mel_corruput_out = np.expand_dims(mel_corruput_out, axis=0)
                mel_corruput_out = torch.from_numpy(mel_corruput_out)
            elif option == 4:
                mel_corruput_out = ir_mel_specgram[:,:,cut_start:cut_end]
            else:
                mel_corruput_out = mel_out
            ret_lis.append(mel_corruput_out)

        return (ret_lis[0], ret_lis[1])