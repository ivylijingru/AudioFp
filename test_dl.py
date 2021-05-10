# test_dl.py
import sys
import os
import numpy as np
import faiss                  # make faiss available

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torch.utils.data import DataLoader as Dataloader
from sklearn import preprocessing

from Model import SimSiam
from DataLoader import SpecData

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

recog_data = SpecData("recognize")
recog_dataloader = Dataloader(recog_data,
                               batch_size = 16,
                               shuffle =False,
                               num_workers = 1)

model.eval()

fingerprint_lis = []
info_query_lis = []

for ii,(spec, info) in enumerate(tqdm(recog_dataloader)):
    embed = model.encode(spec)
    embed = embed.detach().numpy()
    embed = autoL1L2(embed)

    fingerprint_lis.append(embed)
    info_query_lis.append(info)

query = np.asarray(fingerprint_lis)
query = np.vstack(query)

test_data = SpecData("test")

test_dataloader = Dataloader(test_data,
                   batch_size = 16,
                   shuffle =False,
                   num_workers = 1)

import pickle
# test_db_pth = '/home/lijingru/bishe/test_pkl/test_fingerprint.pkl'
test_db_pth = '/home/lijingru/bishe/test_pkl/test_fingerprint_small_402_norm.pkl'

with open(test_db_pth, "rb") as fp:
    fingerprint_bank = pickle.load(fp)

# print(len(fingerprint_bank))
# print(fingerprint_bank[0][0][15])
cvt_lis = []
info_lis = []
# print(len(fingerprint_bank[0]))
for i in range(len(fingerprint_bank)):
    for j in range(len(fingerprint_bank[i][0])):
        info_lis.append((fingerprint_bank[i][1][0][j], fingerprint_bank[i][1][1][j].item()))
        cvt_lis.append(fingerprint_bank[i][0][j])

cvt_np = np.asarray(cvt_lis)
cvt_np = np.vstack(cvt_np)

d = 256
index = faiss.IndexFlatL2(d)   # build the index
index.add(cvt_np)                  # add vectors to the index
# print(index.ntotal)            # 索引中向量的数量。

query_info = []
print(len(info_query_lis))
for i in range(len(info_query_lis)):
    for j in range(len(info_query_lis[i][0])):
        query_info.append((info_query_lis[i][0][j],info_query_lis[i][1][j].item()))

k = 4
D, I = index.search(query, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last qu

cnt = 0
cnt2 = 0

for ii in range(len(query)):
    search_id = I[ii][0]
    search_song = info_lis[search_id]
    real_song = query_info[ii]
    print(search_song, real_song)
    if search_song[0] == real_song[0]:
        cnt += 1
    if search_song[1] == real_song[1]:
        cnt2 += 1

print(cnt, len(query))
print(cnt2, len(query))
