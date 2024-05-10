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


def load_dataset(split="train", SL=30, freq=2):
    data = np.load("../data/handball_shot/{0}.npy".format(split))
    print("Loading {0} split of dataset:".format(split), data.shape)
    return data

class Train(Dataset):

    def __init__(self, data):
        
        self.use_augmentation = True
        self.data = torch.from_numpy(data).to(device)
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        x0, y0 = self.data[idx, 0, :INPUT_LEN], self.data[idx, 0, INPUT_LEN:]
        x1, y1 = self.data[idx, 1, :INPUT_LEN], self.data[idx, 1, INPUT_LEN:]
            
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
                
        if np.random.rand() > 0.5:
            seq0 = seq0[np.arange(-seq0.shape[0]+1, 1)]
            seq1 = seq1[np.arange(-seq1.shape[0]+1, 1)]
            
        return seq0[:INPUT_LEN], seq0[INPUT_LEN:], seq1[:INPUT_LEN], seq1[INPUT_LEN:]

class NotTrain(Dataset):

    def __init__(self, data):

        self.data = torch.from_numpy(data).to(device)

        print("Stats for ctx-num-of-examples:", {0: self.data.shape[0], 1: 0, 2: 0})

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        x0, y0 = self.data[idx, 0, :INPUT_LEN], self.data[idx, 0, INPUT_LEN:]
        x1, y1 = self.data[idx, 1, :INPUT_LEN], self.data[idx, 1, INPUT_LEN:]
        sctx = 0

        sample = {
            'keypoints0': x0.requires_grad_(False).to(device).reshape(1, 16, 13, 3),
            'keypoints1': x1.requires_grad_(False).to(device).reshape(1, 16, 13, 3),
            'out_keypoints0': y0.requires_grad_(False).to(device).reshape(1, 14, 13, 3),
            'out_keypoints1': y1.requires_grad_(False).to(device).reshape(1, 14, 13, 3),
            'sctx': torch.from_numpy(np.array([sctx])).to(device), # 0 - socially dependent, 1 - independent, 2 - same person
        }

        return sample    
    
    
def create_datasets():
    
    pfds = Train(data=load_dataset(split="train"))

    train_loader = DataLoader(pfds, batch_size=256, shuffle=True)

    print("Dataset len: ", len(pfds))

    valid = Train(data=load_dataset(split="valid"))
    test = NotTrain(data=load_dataset(split="test"))
    
    return train_loader, valid, test



