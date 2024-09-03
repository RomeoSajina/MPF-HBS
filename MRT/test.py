import torch
import numpy as np
import torch_dct as dct
import time
from MRT.Models import Transformer
from fncs import VIM
import argparse

import numpy as np
import os
import argparse

from data import TESTDATA

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="3dpw", help='dataset name')
parser.add_argument('--ckp', type=str)
args = parser.parse_args()


test_dataset = TESTDATA(dataset=args.dataset)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device='cpu'

batch_size=1

IN_F = 16
OUT_F = 14
P_DIM = 39


model = Transformer(d_word_vec=128, d_model=128, d_inner=1024,
            n_layers=3, n_head=8, d_k=64, d_v=64,device=device).to(device)

print("# Param:", sum(p.numel() for p in model.parameters() if p.requires_grad))


model.load_state_dict(torch.load(args.ckp, map_location=device)) 

y_pred = []
y_test = []

with torch.no_grad():
    model.eval()
    for jjj,data in enumerate(test_dataloader,0):

        input_seq,output_seq=data
        
        input_seq=torch.tensor(input_seq,dtype=torch.float32).to(device)
        output_seq=torch.tensor(output_seq,dtype=torch.float32).to(device)
        n_joints=int(input_seq.shape[-1]/3)
        use=[input_seq.shape[1]]
        
        input_=input_seq.view(-1,IN_F,input_seq.shape[-1])
  
    
        output_=output_seq.view(output_seq.shape[0]*output_seq.shape[1],-1,input_seq.shape[-1])

        input_ = dct.dct(input_)
        output__ = dct.dct(output_[:,:,:])
        
        
        rec_=model.forward(input_[:,1:IN_F,:]-input_[:,:IN_F-1,:],dct.idct(input_[:,-1:,:]),input_seq,use)
        
        rec=dct.idct(rec_)

        results=output_[:,:1,:]
        for i in range(1,IN_F+1):
            results=torch.cat([results,output_[:,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
        results=results[:,1:,:]
        
        #print("\n\n", output_seq[:, :, 1:].shape, results[:, :OUT_F].unsqueeze(0).shape)
        
        y_pred.append(results[:, :OUT_F].unsqueeze(0).cpu().numpy())
        y_test.append(output_seq[:, :, 1:].cpu().numpy())

y_pred = np.concatenate(y_pred, axis=0)
y_test = np.concatenate(y_test, axis=0)
    
#print(y_test.shape, y_pred.shape)

vims = " ".join( [ str(round(np.mean( [(VIM(pred[0][:LEN], gt[0][:LEN]) + VIM(pred[1][:LEN], gt[1][:LEN])) / 2. for pred, gt in zip(y_pred, y_test)] ) * 100, 1)) for LEN in [2, 4, 8, 10, 14]]  )

print("Test [100ms 240ms 500ms 640ms 900ms]:", vims)
np.save("../data/predictions/{0}_mrt".format(args.dataset), y_pred)
