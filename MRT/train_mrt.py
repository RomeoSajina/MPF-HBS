import os
import torch
import torch.optim as optim
import numpy as np
import torch_dct as dct #https://github.com/zh217/torch-dct
import time

from MRT.Models import Transformer,Discriminator
from utils import disc_l2_loss,adv_disc_l2_loss
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
from fncs import VIM


def calc_vims(ds_loader, model):
    y_pred = []
    y_test = []

    with torch.no_grad():
        model.eval()
        for jjj,data in enumerate(ds_loader,0):

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

            y_pred.append(results[:, :OUT_F].unsqueeze(0).cpu().numpy())
            y_test.append(output_seq[:, :, 1:].cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    #print(y_test.shape, y_pred.shape)

    vims = [np.mean( [(VIM(pred[0][:LEN], gt[0][:LEN]) + VIM(pred[1][:LEN], gt[1][:LEN])) / 2. for pred, gt in zip(y_pred, y_test)] ) * 100 for LEN in [2, 4, 8, 10, 14]]
    return vims, np.mean(vims)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="3dpw", help='dataset name')
args = parser.parse_args()


if not os.path.exists(f'./saved_model/{args.dataset}/'):
    os.makedirs(f'./saved_model/{args.dataset}/', exist_ok=True)


from data import TESTDATA

test_dataset = TESTDATA(dataset=args.dataset)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


from data import DATA
dataset = DATA(args.dataset)
batch_size=64

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

from discriminator_data import D_DATA
real_=D_DATA(args.dataset)

real_motion_dataloader = torch.utils.data.DataLoader(real_, batch_size=batch_size, shuffle=True)
real_motion_all=list(enumerate(real_motion_dataloader))

device='cuda'

IN_F = 16
OUT_F = 14
P_DIM = 39


model = Transformer(d_word_vec=128, d_model=128, d_inner=1024,
            n_layers=3, n_head=8, d_k=64, d_v=64,device=device).to(device)

discriminator = Discriminator(d_word_vec=P_DIM, d_model=P_DIM, d_inner=256,
            n_layers=3, n_head=8, d_k=32, d_v=32,device=device).to(device)


lrate=0.0003
lrate2=0.0005

params = [
    {"params": model.parameters(), "lr": lrate}
]
optimizer = optim.Adam(params)
params_d = [
    {"params": discriminator.parameters(), "lr": lrate}
]
optimizer_d = optim.Adam(params_d)


best_vim_avg = 500
    
for epoch in range(100):
    total_loss=0
    
    for j,data in enumerate(dataloader,0):
                
        use=None
        input_seq,output_seq=data
        input_seq=torch.tensor(input_seq,dtype=torch.float32).to(device) # batch, N_person, 15 (15 fps 1 second), 45 (15joints xyz) 
        output_seq=torch.tensor(output_seq,dtype=torch.float32).to(device) # batch, N_persons, 46 (last frame of input + future 3 seconds), 45 (15joints xyz) 
        
        # first 1 second predict future 1 second
        input_=input_seq.view(-1,IN_F,input_seq.shape[-1]) # batch x n_person ,15: 15 fps, 1 second, 45: 15joints x 3
        
        output_=output_seq.view(output_seq.shape[0]*output_seq.shape[1],-1,input_seq.shape[-1])

        input_ = dct.dct(input_)
                
        rec_=model.forward(input_[:,1:IN_F,:]-input_[:,:IN_F-1,:],dct.idct(input_[:,-1:,:]),input_seq,use)

        rec=dct.idct(rec_)
        
        results=output_[:,:1,:]
        for i in range(1,OUT_F+1):
            results=torch.cat([results,output_[:,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
        results=results[:,1:,:]
          
        #print(rec.shape, output_.shape)
            
        loss=torch.mean((rec[:,:OUT_F,:]-(output_[:,1:OUT_F+1,:]-output_[:,:OUT_F,:]))**2)
        
        #print("loss:", loss)
        
        if (j+1)%2==0:
            torch.autograd.set_detect_anomaly(True)
            fake_motion=results

            #disc_loss=disc_l2_loss(discriminator(fake_motion))
            #loss=loss+0.0005*disc_loss

            fake_motion=fake_motion.detach()

            real_motion=real_motion_all[int(j/2)][1][1]
            real_motion=real_motion.view(-1,OUT_F+1,P_DIM)[:,1:OUT_F+1,:].float().to(device)

            fake_disc_value = discriminator(fake_motion)
            real_disc_value = discriminator(real_motion)

            d_motion_disc_real, d_motion_disc_fake, d_motion_disc_loss = adv_disc_l2_loss(real_disc_value, fake_disc_value)

            optimizer_d.zero_grad()
            d_motion_disc_loss.backward(retain_graph=True)
            optimizer_d.step()
            
            disc_loss=disc_l2_loss(discriminator(fake_motion))
            loss=loss+0.0005*disc_loss

       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        total_loss=total_loss+loss

    print('epoch:',epoch,'loss:',total_loss/(j+1))
    if (epoch+1)%5==0:
        save_path=f'./saved_model/{args.dataset}/{epoch}.model'
        torch.save(model.state_dict(),save_path)
        
    
    vims, test_vim_avg = calc_vims(test_dataloader, model)
    if best_vim_avg > test_vim_avg:
        best_vim_avg = test_vim_avg
        torch.save(model.state_dict(), f'./saved_model/{args.dataset}/best.model')


        
