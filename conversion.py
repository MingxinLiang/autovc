# condig: utf-8
import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator
from ipdb import set_trace


def pad_seq(x, base=32):
    """ 按base数并行，计算输出长度，补齐
    @return:
        x: 补齐后的音频
        len_pad, 补齐的帧
    """
    len_out = int(base * ceil(float(x.shape[0])/base)) #
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

# set generater 
# in evaluation model
device = 'cuda:0'
G = Generator(32,256,512,32).eval().to(device)
g_checkpoint = torch.load('autovc.ckpt', map_location='cuda:0')
G.load_state_dict(g_checkpoint['model'])
# matada: list, audio data, [[name, emb, voice2],...]
# emb: np.array, [emb_dim]
# voice: np.array, [length, mel_dim]
metadata = pickle.load(open('metadata.pkl', "rb"))
set_trace()
spect_vc = []

for sbmt_i in metadata:
             
    x_org = sbmt_i[2]
    x_org, len_pad = pad_seq(x_org)
    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)

    for sbmt_j in metadata:
                   
        emb_trg = torch.from_numpy(sbmt_j[1][np.newaxis, :]).to(device)
        
        with torch.no_grad():
            _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
            
        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
        
        spect_vc.append( ('{}x{}'.format(sbmt_i[0], sbmt_j[0]), uttr_trg) )
        
        
with open('results.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)          