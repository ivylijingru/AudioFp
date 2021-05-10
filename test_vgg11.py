import argparse
import sys
import os
import pickle
import faiss

import torch
import copy
import numpy as np

from Model import SimSiam
from DataLoader_1channel import SpecData
from tqdm.auto import tqdm
from torch.utils.data import DataLoader as Dataloader
from sklearn import preprocessing

def autoL1L2(data, norms = 'l2'):
    '''L1或者L2正则化'''
    return preprocessing.normalize(data, norm = norms)

def ensureFolderExists(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)

def stackFingerprint(fingerprint_bank):
    fingerprint_bank = np.asarray(fingerprint_bank)
    fingerprint_bank = np.vstack(fingerprint_bank)
    print(fingerprint_bank.shape)
    return fingerprint_bank

def stackInfo(info_lis):
    info = []
    for i in range(len(info_lis)):
        for j in range(len(info_lis[i][0])):
            info.append((info_lis[i][0][j],info_lis[i][1][j].item()))
    return info

def readPkl(pklPath):
    with open(pklPath, "rb") as fp:
        lis = pickle.load(fp)   
    return lis

def savePkl(lis, pklPath):
    with open(pklPath, "wb") as fp:
        pickle.dump(lis, fp)

def fingerprinter(model, fpPath, infoPath):
    if not os.path.exists(fpPath) and not os.path.exists(infoPath):        
        test_data = SpecData("test")
        test_dataloader = Dataloader(test_data,
                                    batch_size = 16,
                                    shuffle =False,
                                    num_workers = 0)

        fingerprint_bank = []
        info_lis = []
        # cnt = 0
        for (spec, info) in tqdm(test_dataloader):
            # cnt += 1
            # if cnt == 10:
            #     break
            embed = model.encode(spec)
            
            embed = embed.detach().numpy()
            embed = autoL1L2(embed)
            fingerprint_bank.append(embed)
            info_lis.append(info)
        # save for visualization afterwards
        savePkl(fingerprint_bank, fpPath)
        savePkl(info_lis, infoPath)
    else:
        fingerprint_bank = readPkl(fpPath)
        info_lis = readPkl(infoPath)

    return fingerprint_bank, info_lis

def prepareTestFp(model, snr, checkPath):
    recog_data = SpecData("recognize", snr=snr)
    recog_dataloader = Dataloader(recog_data,
                                batch_size = 16,
                                shuffle =False,
                                num_workers = 0)

    query_fp = []
    info_query_lis = []
    # cnt = 0
    for (spec, info) in tqdm(recog_dataloader):
        # cnt += 1
        # if cnt == 10:
        #     break
        embed = model.encode(spec)
        embed = embed.detach().numpy()
        embed = autoL1L2(embed)

        query_fp.append(embed)
        info_query_lis.append(info)

    savePkl(info_query_lis, checkPath)
    return query_fp, info_query_lis

def recognizer(dbFpPair, testFpPair, resPath):
    fingerprint_bank, info_lis = dbFpPair
    query_fp, info_query_lis = testFpPair
    info_lis = stackInfo(info_lis)
    info_query_lis = stackInfo(info_query_lis)

    d = 256
    index = faiss.IndexFlatL2(d)   # build the index
    index.add(stackFingerprint(fingerprint_bank))
    k = 4
    D, I = index.search(stackFingerprint(query_fp), k)
    
    cnt1 = 0
    cnt2 = 0
    
    for ii in range(len(info_query_lis)):
        search_id = I[ii][0]
        search_song = info_lis[search_id]
        real_song = info_query_lis[ii]
        # print(search_song, real_song)
        if search_song[0] == real_song[0]:
            cnt1 += 1
        if search_song[0] == real_song[0] and search_song[1] == real_song[1]:
            cnt2 += 1
    
    with open(resPath, 'w') as fp:
        fp.write(str(cnt1) + ' ' + str(cnt2) + ' ' + str(len(info_query_lis)))
    
if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser(description='Create fingerprint database and test accuracy')
    parser.add_argument('snr', type=str, help='3, 10, 20')
    parser.add_argument('modelPath', type=str, help='Path of the model.')
    parser.add_argument('saveDir', type=str, help='Path to save the result.')

    args = parser.parse_args()
    
    model = SimSiam(
        latent_dim=256,
        proj_hidden_dim=256,
        pred_hidden_dim=128
    )
    
    # model.load_state_dict(args.modelPath)
    model.encoder.load_state_dict(torch.load(args.modelPath))
    # device = torch.device('cpu')
    # model = model.to(device)
    # model.load_state_dict(torch.load(args.modelPath))
    # model.load_state_dict(torch.load(args.modelPath), map_location=device)
    model.eval()

    save_time = args.modelPath.split('/')[-2].split('.')[0]
    epoch = args.modelPath.split('/')[-1].split('.')[0].split('_')[1]

    rt_dir = os.path.join(args.saveDir, save_time + '_' + epoch)
    fpPath = os.path.join(rt_dir, 'fingerprint.pkl')
    infoPath = os.path.join(rt_dir, 'info.pkl')
    resPath = os.path.join(rt_dir, args.snr + '_' + 'res.txt')
    checkPath = os.path.join(rt_dir, 'debug.pkl')

    ensureFolderExists(args.saveDir)
    ensureFolderExists(rt_dir)

    dbFpPair = fingerprinter(model, fpPath, infoPath)
    testFpPair = prepareTestFp(model, args.snr, checkPath)

    recognizer(dbFpPair, testFpPair, resPath)