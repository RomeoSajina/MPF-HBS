import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np
import torch_dct as dct
from models.net import TBIFormer
from tqdm import tqdm
from utils.opt import Options
from utils.dataloader import Data
from utils.metrics import FDE, JPE, APE
from utils.TRPE import bulding_TRPE_matrix
from fncs import VIM



if __name__ == '__main__':
    opt = Options().parse()
    device = opt.device
    test_dataset = Data(dataset=opt.dataset, mode=-1, transform=False, device=device, opt=opt)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)


    model = TBIFormer(input_dim=opt.d_model, d_model=opt.d_model,
                        d_inner=opt.d_inner, n_layers=opt.num_stage,
                        n_head=opt.n_head , d_k=opt.d_k, d_v=opt.d_v, dropout=opt.dropout, device=device,kernel_size=opt.kernel_size, opt=opt).to(device)



    checkpoint = torch.load(opt.ckp, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model loaded.')
    #print(model)
    print(f"best_epoch: {checkpoint['epoch']}")
    
    y_test = []
    y_pred = []
    with torch.no_grad():
        model.eval()
        loss_list=[]

        n = 0

        for batch_i, batch_data in tqdm(enumerate(test_dataloader, 0)):
            n+=1
            input_seq, output_seq = batch_data
            B, N, _, D = input_seq.shape
            input_ = input_seq.view(-1, opt.input_time, input_seq.shape[-1])
            output_ = output_seq.view(output_seq.shape[0] * output_seq.shape[1], -1, input_seq.shape[-1])


            trj_dist = bulding_TRPE_matrix(input_seq.reshape(B,N,-1,opt.n_joints,3), opt)  #  trajectory similarity distance
            offset = input_[:, 1:opt.input_time, :] - input_[:, :opt.input_time-1, :]     #   dispacement sequence
            src = dct.dct(offset)

            rec_ = model.forward(src, N, trj_dist)
            rec = dct.idct(rec_)
            results = output_[:, :1, :]
            for i in range(1, opt.output_time+1):
                results = torch.cat(
                    [results, output_[:, :1, :] + torch.sum(rec[:, :i, :], dim=1, keepdim=True)],
                    dim=1)
            results = results[:, 1:, :]  # 3 15 45

            prediction = results.view(B, N, -1, opt.n_joints, 3)
            gt = output_.view(B, N, -1, opt.n_joints, 3)

            #print("\n\n", gt.shape, prediction.shape)
            y_test.append(gt[:, :, -14:].cpu().numpy())
            y_pred.append(prediction[:, :, -14:].cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    #print(y_test.shape, y_pred.shape)

    vims = " ".join( [ str(round(np.mean( [(VIM(pred[0][:LEN], gt[0][:LEN]) + VIM(pred[1][:LEN], gt[1][:LEN])) / 2. for pred, gt in zip(y_pred, y_test)] ) * 100, 1)) for LEN in [2, 4, 8, 10, 14]]  )

    print("Test [100ms 240ms 500ms 640ms 900ms]:", vims)
    np.save("../data/predictions/{0}_tbiformer".format(opt.dataset), y_pred)
