#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional
import numpy as np
from progress.bar import Bar
import pandas as pd

from utils import loss_funcs, utils as utils
from utils.opt import Options
from utils.handball_shot3d import DS3D
import utils.model as nnmodel
import utils.data_utils as data_utils


def main(opt):

    # create model
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    dct_n = opt.dct_n
    is_cuda = torch.cuda.is_available()

    model = nnmodel.GCN(input_feature=dct_n, hidden_feature=opt.linear_size, p_dropout=opt.dropout,
                        num_stage=opt.num_stage, node_n=39)

    if is_cuda:
        model.cuda()

    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    model_path_len = opt.ckpt + opt.ckpt_name
    
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    if is_cuda:
        ckpt = torch.load(model_path_len)
    else:
        ckpt = torch.load(model_path_len, map_location='cpu')
    start_epoch = ckpt['epoch']
    err_best = ckpt['err']
    lr_now = ckpt['lr']
    model.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    
    # data loading
    print(">>> loading data")
    test_dataset = DS3D(path_to_data=opt.data_dir_3dpw, input_n=input_n, output_n=output_n, split=1,
                              dct_n=dct_n)
    dim_used = test_dataset.dim_used
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=180,#opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)
    print(">>> test data {}".format(test_dataset.__len__()))

    test_3d = test(test_loader, model,
                   input_n=input_n,
                   output_n=output_n,
                   is_cuda=is_cuda,
                   dim_used=dim_used,
                   dct_n=dct_n)



def test(train_loader, model, input_n=20, output_n=50, is_cuda=False, dim_used=[], dct_n=15):
    N = 0
    if output_n == 15:
        eval_frame = [2, 5, 8, 11, 14]
    elif output_n == 30:
        eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
    elif output_n == 14:
        eval_frame = [1, 3, 7, 9, 13]
    t_3d = np.zeros(len(eval_frame))

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
        
    y_preds = []
    y_tests = []
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            # targets = Variable(targets.cuda(async=True)).float()
            all_seq = Variable(all_seq.cuda(non_blocking=True)).float()
        else:
            inputs = Variable(inputs).float()
            # targets = Variable(targets).float()
            all_seq = Variable(all_seq).float()
        outputs = model(inputs)

        n, seq_len, dim_full_len = all_seq.data.shape

        _, idct_m = data_utils.get_dct_matrix(seq_len)
        idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
        outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
        outputs_exp = torch.matmul(idct_m[:, 0:dct_n], outputs_t).transpose(0, 1).contiguous().view \
            (-1, dim_full_len , seq_len).transpose(1, 2) # dim_full_len- 3
        
        ## 
        yp = outputs_exp[:, -14:].cpu().detach().numpy().reshape(-1, 2, 14, 39)
        yt = all_seq[:, -14:].cpu().detach().numpy().reshape(-1, 2, 14, 39)
        y_preds.append(yp)
        y_tests.append(yt)
        ###
       
        pred_3d = all_seq.clone()
        pred_3d[:, :, dim_used] = outputs_exp
        pred_p3d = pred_3d.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]
        targ_p3d = all_seq.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]
        
        for k in np.arange(0, len(eval_frame)):
            j = eval_frame[k]
            t_3d[k] += torch.mean(torch.norm(
                targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3), 2,
                1)).cpu().data.numpy() * n

        # update the training loss
        N += n

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    
    ####
    y_pred = np.concatenate(y_preds, axis=0)
    y_test = np.concatenate(y_tests, axis=0)
    
    vims = " ".join( [ str(round(np.mean( [(VIM(pred[0][:LEN], gt[0][:LEN]) + VIM(pred[1][:LEN], gt[1][:LEN])) / 2. for pred, gt in zip(y_pred, y_test)] ) * 100, 1)) for LEN in [2, 4, 8, 10, 14]]  )

    print("Test [100ms 240ms 500ms 640ms 900ms]:", vims)
    np.save("../data/predictions/handball_shot_ltd", y_pred)
    ####
    return t_3d / N


def VIM(pred, GT, calc_per_frame=True, return_last=True):
    if calc_per_frame:
        pred = pred.reshape(-1, 39)
        GT = GT.reshape(-1, 39)
    errorPose = np.power(GT - pred, 2)
    errorPose = np.sum(errorPose, 1)
    errorPose = np.sqrt(errorPose)
    
    if return_last:
        errorPose = errorPose[-1]
    return errorPose


if __name__ == "__main__":
    option = Options().parse()
    main(option)
