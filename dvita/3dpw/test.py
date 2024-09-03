import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import utils
import time
import argparse
import json

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

def main(args):
    import json           

    ######################loading data#######################
    dev = args.dev 
    #dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    import DataLoader_test
    val = DataLoader_test.data_loader(args)

    if args.dataset == "3dpw":
        with open("../../data/somof/3dpw_test_out.json") as f:
            test_gt = np.array(json.load(f))
    elif args.dataset == "handball_shot":
        test_gt = val.dataset.data_out
    
    
    ####################defining model#######################
    import model 
    net_g = model.LSTM_g(embedding_dim=args.embedding_dim, h_dim=args.hidden_dim, dropout=args.dropout, dev=dev).to(device=dev)
    encoder = model.Encoder(h_dim=args.hidden_dim, latent_dim=args.latent_dim, dropout=args.dropout, dev=dev)
    decoder = model.Decoder(h_dim=args.hidden_dim, latent_dim=args.latent_dim, dropout=args.dropout, dev=dev)
    net_l = model.VAE(Encoder=encoder, Decoder=decoder).to(device=dev)
    net_l.double()
    net_g.double()
    
    print("# Param:", sum(p.numel() for p in net_l.parameters() if p.requires_grad) + sum(p.numel() for p in net_g.parameters() if p.requires_grad))

    ########################load params#########################
    #net_g.load_state_dict(torch.load("checkpoint_g.pkl"))
    #net_l.load_state_dict(torch.load("checkpoint_l.pkl"))
    print("loading", args.ckp, "....")
    net_g.load_state_dict(torch.load("./models/{0}/checkpoint_g_{1}.pkl".format(args.model_folder, args.ckp)))
    net_l.load_state_dict(torch.load("./models/{0}/checkpoint_l_{1}.pkl".format(args.model_folder, args.ckp)))

    net_g.eval()
    net_l.eval()
    for idx, (obs_p, obs_s, obs_f, target_f, start_end_idx) in enumerate(val):
    
        batch = obs_p.size(1) 
        obs_p = obs_p.to(device=dev).double()
        obs_s = obs_s.to(device=dev).double()
        
        #########splitting the motion into local + global##########
        obs_s_g = 0.5*(obs_s.view(15, batch, 13, 3)[:,:,0] + obs_s.view(15, batch, 13, 3)[:,:,1])
        obs_s_l = (obs_s.view(15, batch, 13, 3) - obs_s_g.view(15, batch, 1, 3)).view(15, batch, 39)
        ###########################################################
        with torch.no_grad():
            #####predicting the global speed and calculate mse loss####
            speed_preds_g = net_g(global_s=obs_s_g)
            ######predicting the local speed using VAE and calculate loss########### 
            output, mean, log_var = net_l(obs_s_l)
            ###########################################################
            speed_preds = (speed_preds_g.view(14, batch, 1, 3) + output.view(14, batch, 13, 3)).view(14, batch, 39)
            ##################calculating the predictions#######################
            preds_p = utils.speed2pos(speed_preds, obs_p, dev=dev) 

            alist = []
            for _, (start, end) in enumerate(start_end_idx):
                alist.append(preds_p.permute(1,0,2)[start:end].tolist())
            with open('{0}_predictions.json'.format(args.dataset), 'w') as f:
                f.write(json.dumps(alist))
        #####################calculating the metrics########################
        y_pred = np.array(alist)
        vims = " ".join( [ str(round(np.mean( [(VIM(pred[0][:LEN], gt[0][:LEN]) + VIM(pred[1][:LEN], gt[1][:LEN])) / 2. for pred, gt in zip(y_pred, test_gt)] ) * 100, 1)) for LEN in [2, 4, 8, 10, 14]]  )

        print("Test [100ms 240ms 500ms 640ms 900ms]:", vims)
        np.save("../../data/predictions/{0}_dvita".format(args.dataset), y_pred)
        
    print('Done !')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default="3dpw", type=str, required=False)
    
    parser.add_argument('--hidden_dim', default=64, type=int, required=False)
    parser.add_argument('--latent_dim', default=32, type=int, required=False)
    parser.add_argument('--embedding_dim', default=8, type=int, required=False)
    parser.add_argument('--dropout', default=0., type=float, required=False)
    parser.add_argument('--ckp', default='best_epoch', type=str, required=False)
    parser.add_argument('--model_folder', default='3dpw', type=str, required=False)
    parser.add_argument('--dev', default='cpu', type=str, required=False)
    args = parser.parse_args()

    main(args)

