from utils import *
import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
seed = 1111
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")


INPUT_LEN = 16
OUTPUT_LEN = 14


DATA_PATH = "../data/"

def l(name):
    with open(DATA_PATH + "somof/3dpw_{0}_in.json".format(name)) as f:
        X = np.array(json.load(f))

    with open(DATA_PATH + "somof/3dpw_{0}_out.json".format(name)) as f:
        Y = np.array(json.load(f))

    X = X if X.shape[-1] == 3 else X.reshape(*X.shape[:-1], 13, 3)
    Y = Y if Y.shape[-1] == 3 else Y.reshape(*Y.shape[:-1], 13, 3)
    
    XY = np.concatenate((X, Y), axis=2)
    X, Y = XY[:, :, :16], XY[:, :, 16:]
    
    print(X.shape, Y.shape)
    data = [
        {
        'keypoints0': torch.from_numpy(x[0].reshape(1, 16, 13, 3)).requires_grad_(False).to(device),
        'keypoints1': torch.from_numpy(x[1].reshape(1, 16, 13, 3)).requires_grad_(False).to(device),
        'out_keypoints0': torch.from_numpy(y[0].reshape(1, 14, 13, 3)).requires_grad_(False).to(device),
        'out_keypoints1': torch.from_numpy(y[1].reshape(1, 14, 13, 3)).requires_grad_(False).to(device),            
        }
      for x, y in zip(X, Y)]
    
    return data



class PoseForecastingDS3DPWAmass(Dataset):

    def __init__(self, x3dpw, x3dpw_single, xamass):
        
        self.use_augmentation = True
        self.xamass = torch.from_numpy(xamass).to(device)
        self.x3dpw = torch.from_numpy(x3dpw).to(device)
        self.x3dpw_single = torch.from_numpy(x3dpw_single).to(device)
        
    def __len__(self):
        return self.x3dpw.shape[0] + self.x3dpw_single.shape[0] + self.xamass.shape[0]

    def __getitem__(self, idx):
        
        if idx < self.x3dpw.shape[0]:
            x0, y0 = self.x3dpw[idx, 0, :INPUT_LEN], self.x3dpw[idx, 0, INPUT_LEN:]
            x1, y1 = self.x3dpw[idx, 1, :INPUT_LEN], self.x3dpw[idx, 1, INPUT_LEN:]

        elif idx < self.x3dpw.shape[0] + self.x3dpw_single.shape[0]:    
            idx -= self.x3dpw.shape[0]
            partner_idx = np.random.randint(0, self.x3dpw_single.shape[0])
            x0, y0 = self.x3dpw_single[idx, 0, :INPUT_LEN], self.x3dpw_single[idx, 0, INPUT_LEN:]
            x1, y1 = self.x3dpw_single[partner_idx, 0, :INPUT_LEN], self.x3dpw_single[partner_idx, 0, INPUT_LEN:]
            
        elif idx < self.x3dpw.shape[0] + self.x3dpw_single.shape[0] + self.xamass.shape[0]:
            idx -= self.x3dpw.shape[0] + self.x3dpw_single.shape[0]
            partner_idx = np.random.randint(0, self.xamass.shape[0])
            x0, y0 = self.xamass[idx, 0, :INPUT_LEN], self.xamass[idx, 0, INPUT_LEN:]
            x1, y1 = self.xamass[partner_idx, 0, :INPUT_LEN], self.xamass[partner_idx, 0, INPUT_LEN:]

            
        if self.use_augmentation:
            x0, y0, x1, y1 = self._augment(x0, y0, x1, y1)
            
            
        sample = {
            'keypoints0': x0.requires_grad_(False).to(device),
            'keypoints1': x1.requires_grad_(False).to(device), 
            'out_keypoints0': y0.requires_grad_(False).to(device),
            'out_keypoints1': y1.requires_grad_(False).to(device),
        }
      
        return sample
    
    def _augment(self, x0, y0, x1, y1):
        
        if np.random.rand() > 0.5:
            return x0, y0, x1, y1
        
        seq0 = torch.cat((x0, y0), dim=0)
        seq1 = torch.cat((x1, y1), dim=0)
                
        if np.random.rand() > 0.5: # backward movement, is this flip?? torch.flip(seq0, dim=0)
            seq0 = seq0[np.arange(-seq0.shape[0]+1, 1)]
            seq1 = seq1[np.arange(-seq1.shape[0]+1, 1)]
            
        return seq0[:INPUT_LEN], seq0[INPUT_LEN:], seq1[:INPUT_LEN], seq1[INPUT_LEN:]

def create_datasets():
    
    x_amass = torch.from_numpy(np.load(DATA_PATH + "amass/amass-cmu_blm_troje.npy", allow_pickle=True)) 

    xy2, xy1 = [], []
    x2, y2, x1, y1 = load_original_3dw(input_window=INPUT_LEN, output_window=OUTPUT_LEN, frequency=2)
    xy2.append(np.concatenate((x2, y2), axis=2))
    xy1.append(np.concatenate((x1, y1), axis=2))

    pfds = PoseForecastingDS3DPWAmass(x3dpw=np.concatenate(xy2, axis=0), x3dpw_single=np.concatenate(xy1, axis=0), xamass=x_amass.reshape(-1, 1, INPUT_LEN+OUTPUT_LEN, 13, 3).numpy())

    train_loader = DataLoader(pfds, batch_size=256, shuffle=True)

    print("Dataset len: ", len(pfds))

    valid = l("valid")
    test = l("test")
    
    return train_loader, valid, test



