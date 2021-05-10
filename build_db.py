# create database
import sys
import os
import pickle
import torch
import copy
import numpy as np

from Model import SimSiam
from DataLoader import SpecData
from tqdm.auto import tqdm
from torch.utils.data import DataLoader as Dataloader
from sklearn import preprocessing

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def autoL1L2(data, norms = 'l2'):
    '''L1或者L2正则化'''
    return preprocessing.normalize(data, norm = norms)

model = SimSiam(
    latent_dim=256,
    proj_hidden_dim=256,
    pred_hidden_dim=128
)

# pth = os.path.join('/home/lijingru/bishe/JupyterNote/runs/Apr08_12-55-50_localhost.localdomain',  'Simsiam_' + str(23) + ".pt")
# pth = os.path.join('/home/lijingru/bishe/Experiment/runs/Apr22_01-41-31_localhost.localdomain',  'Simsiam_' + str(13) + ".pt")
pth = os.path.join('/home/lijingru/bishe/Experiment/runs/Apr28_19-47-19_localhost.localdomain',  'Simsiam_' + str(402) + ".pt")
model.encoder.load_state_dict(torch.load(pth))
model.eval()

test_data = SpecData("test")
test_dataloader = Dataloader(test_data,
                   batch_size = 16,
                   shuffle =False,
                   num_workers = 1)

fingerprint_bank = []
resample = []

for ii,(spec, info) in enumerate(tqdm(test_dataloader)):
    embed = model.encode(spec)
    embed = embed.detach().numpy()
    embed = autoL1L2(embed)
    fingerprint_bank.append((embed, info))

# 这在写一些什么东西。。。
fingerprint_bank = np.asarray(fingerprint_bank)
fingerprint_bank = np.vstack(fingerprint_bank)

test_db_pth = '/home/lijingru/bishe/test_pkl/test_fingerprint_small_402_norm.pkl'
with open(test_db_pth, "wb") as fp:
    pickle.dump(fingerprint_bank, fp)
