import os
import torch
import torch.optim as optim
from model import Futuremotion_ICCV21
from utils import VIM
import numpy as np
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', 
                    type=str, 
                    default="3dpw",
                    help='dataset name')
parser.add_argument('--ckp', type=str)

args = parser.parse_args()


if not os.path.exists("./models/"+args.dataset):
    os.makedirs("./models/"+args.dataset, exist_ok=True)

if args.dataset == "3dpw":
    from dataset_3dpw import create_datasets
else:
    from dataset_handball_shot import create_datasets

train_loader, valid, test = create_datasets()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Futuremotion_ICCV21().to(device)
print("# Param:", sum(p.numel() for p in model.parameters() if p.requires_grad))

if args.ckp and len(args.ckp) > 0:
    print(">>> Loading model from", args.ckp)
    model.load_state_dict(torch.load(args.ckp))
    model.eval()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def ohkm(loss, top_k):
    loss = torch.mean(loss, dim=1)
    
    ohkm_loss = 0.
    batch_size = loss.shape[0]
    
    for i in range(batch_size):
        sub_loss = loss[i]
        topk_val, topk_idx = torch.topk(sub_loss, k=top_k, sorted=False)
        #print("gather", sub_loss, topk_idx)
        tmp_loss = torch.gather(sub_loss, dim=0, index=topk_idx) # can be ignore ???
        ohkm_loss += torch.sum(tmp_loss) / top_k
    ohkm_loss /= batch_size
    return ohkm_loss

def loss_fn(output, target, train=False):

    if len(target.shape) == 3:
        output = output.unsqueeze(0)
        target = target.unsqueeze(0)
        
    loss = ohkm(torch.norm((output - target), p=2, dim=-1), 6)
    return loss


def calc_vim(ds, model):
    y_pred = []
    y_test = []
    for i, inp in enumerate(ds):
        with torch.no_grad():

            out = model(inp, False)

        p0 = out["z0"][:, :14].reshape(1, 14, 13, 3).float().detach().cpu().numpy()
        p1 = out["z1"][:, :14].reshape(1, 14, 13, 3).float().detach().cpu().numpy()

        del out["z0"], out["z1"]

        y_pred.append(np.concatenate((p0, p1), axis=0))
        y_test.append(np.concatenate((inp["out_keypoints0"].float().detach().cpu().numpy(), inp["out_keypoints1"].float().detach().cpu().numpy()), axis=0))
        print_wei = False

    y_pred = np.array(y_pred)
    vims = [ np.mean( [(VIM(pred[0][:LEN], gt[0][:LEN]) + VIM(pred[1][:LEN], gt[1][:LEN])) / 2. for pred, gt in zip(y_pred, y_test)] ) * 100 for LEN in [2, 4, 8, 10, 14]] 

    return vims, np.mean(vims)


TLoss = 14

def train_one_epoch(epoch_index, TLoss):
    running_loss = 0.
    last_loss = 0.

    for i, x_orig in enumerate(iter(train_loader)):

        x = x_orig.copy()
        
        optimizer.zero_grad(set_to_none=True)
        
        outputs = model(x, True)

        loss = (loss_fn(outputs["z0"].float()[:, :TLoss], x["out_keypoints0"].float()[:, :TLoss], True) + 
                loss_fn(outputs["z1"].float()[:, :TLoss], x["out_keypoints1"].float()[:, :TLoss], True)) / 2.
        
        del outputs
        
        running_loss += loss.item()

        loss.backward()

        optimizer.step()

    avg_loss = running_loss / (i+1)
   
    return avg_loss, 0, 0


EPOCHS = 50
best_vim_avg = 500

for epoch_number in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    model.train(True)
    avg_loss, avg_aux_acc, avg_aux_acc2 = train_one_epoch(epoch_number, TLoss)

    for g in optimizer.param_groups:
        g['lr'] *= 0.95

    if epoch_number%2 == 0:
        TLoss += 1
    
    model.train(False)

    running_vloss = 0.0
    
    for i, vx in enumerate(valid):
        with torch.no_grad():
            voutputs = model(vx, False)
        vloss = (loss_fn(voutputs["z0"][:, :14].reshape(*vx["out_keypoints0"].shape).float(), vx["out_keypoints0"].float()) + 
                 loss_fn(voutputs["z1"][:, :14].reshape(*vx["out_keypoints1"].shape).float(), vx["out_keypoints1"].float())) / 2.

        running_vloss += vloss
        del voutputs["z0"], voutputs["z1"]

    avg_vloss = running_vloss / (i + 1)
    
    print('LOSS train: {0:.4f} valid: {1:.4f} - lr: {2:.4f}'.format(avg_loss, avg_vloss, get_lr(optimizer)))

    epoch_number += 1
    
    _, test_vim_avg = calc_vim(test, model)
    if best_vim_avg > test_vim_avg:
        best_vim_avg = test_vim_avg
        torch.save(model.state_dict(), "./models/{0}/futuremotion_iccv2021_best.pt".format(args.dataset))    

import datetime
torch.save(model.state_dict(), "./models/{0}/futuremotion_iccv2021_{1}.pt".format(args.dataset, datetime.datetime.now().timestamp()))