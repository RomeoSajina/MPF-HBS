import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os

from dataset.data_utils import *


def load_dataset(split="train", SL=30, freq=2):
    data = np.load("../data/handball_shot/{0}.npy".format(split))
    print("Loading {0} split of dataset:".format(split), data.shape)
    return data


class TrainDS(Dataset):

    def __init__(self, config, data):
        self.config = config
        self.use_augmentation = True
        self.data = torch.from_numpy(data).to(config.device)

        print("Stats for TrainDS ctx-num-of-examples:", { 0: self.data.shape[0], 1: 0, 2: 0})

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        INPUT_LEN = self.config.input_len

        if idx < self.data.shape[0]:
            x0, y0 = self.data[idx, 0, :INPUT_LEN], self.data[idx, 0, INPUT_LEN:]
            x1, y1 = self.data[idx, 1, :INPUT_LEN], self.data[idx, 1, INPUT_LEN:]
            sctx = 0

        if self.use_augmentation:
            x0, y0, x1, y1, sctx = self._augment(x0, y0, x1, y1, sctx)

        sample = {
            'keypoints0': x0.requires_grad_(False).to(self.config.device),
            'keypoints1': x1.requires_grad_(False).to(self.config.device),
            'out_keypoints0': y0.requires_grad_(False).to(self.config.device),
            'out_keypoints1': y1.requires_grad_(False).to(self.config.device),
            'sctx': torch.from_numpy(np.array([sctx])).to(self.config.device), # 0 - socially dependent, 1 - independent, 2 - same person
        }

        return sample

    def _augment(self, x0, y0, x1, y1, sctx):

        if np.random.rand() > 0.5:
            return x0, y0, x1, y1, sctx

        seq0 = torch.cat((x0, y0), dim=0)
        seq1 = torch.cat((x1, y1), dim=0)

        if self.config.augment.backward_movement and np.random.rand() > 0.5: # backward movement, is this flip?? torch.flip(seq0, dim=0)
            seq0 = seq0[np.arange(-seq0.shape[0]+1, 1)]
            seq1 = seq1[np.arange(-seq1.shape[0]+1, 1)]

        if self.config.augment.reversed_order and np.random.rand() > 0.5: # reversed order of people
            seq0, seq1 = seq1, seq0

        if self.config.augment.random_scale and np.random.rand() > 0.5: # random scale
            r1=0.1#0.8
            r2=5.#1.2
            def _rand_scale(_x):
                if np.random.rand() > 0.5:
                    rnd = ((r1 - r2) * np.random.rand() + r2)

                    scld = _x * rnd
                    scld += (_x[:, 7] - scld[:, 7]).reshape(-1, 1, 3) # restore global position, TODO: scaled-motion
                    return scld
                return _x
            seq0 = _rand_scale(seq0)
            seq1 = _rand_scale(seq1)

        if self.config.augment.random_rotate_y and np.random.rand() > 0.75:
            seq0, seq1 = random_rotate_sequences(seq0, seq1, rotate_around="y", device=self.config.device)
        if self.config.augment.random_rotate_x and np.random.rand() > 0.75:
            seq0, seq1 = random_rotate_sequences(seq0, seq1, rotate_around="x", device=self.config.device)
        if self.config.augment.random_rotate_z and np.random.rand() > 0.75:
            seq0, seq1 = random_rotate_sequences(seq0, seq1, rotate_around="z", device=self.config.device)

        if self.config.augment.random_reposition and np.random.rand() > 0.5:
            seq0, seq1 = random_reposition_sequences(seq0, seq1, device=self.config.device)
        
        return seq0[:self.config.input_len], seq0[self.config.input_len:], seq1[:self.config.input_len], seq1[self.config.input_len:], sctx


class DS(Dataset):

    def __init__(self, config, data):

        self.config = config

        self.data = torch.from_numpy(data).to(config.device)

        print("Stats for DS ctx-num-of-examples:", {0: self.data.shape[0], 1: 0, 2: 0})

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        INPUT_LEN = self.config.input_len

        x0, y0 = self.data[idx, 0, :INPUT_LEN], self.data[idx, 0, INPUT_LEN:]
        x1, y1 = self.data[idx, 1, :INPUT_LEN], self.data[idx, 1, INPUT_LEN:]
        sctx = 0

        sample = {
            'keypoints0': x0.requires_grad_(False).to(self.config.device),
            'keypoints1': x1.requires_grad_(False).to(self.config.device),
            'out_keypoints0': y0.requires_grad_(False).to(self.config.device),
            'out_keypoints1': y1.requires_grad_(False).to(self.config.device),
            'sctx': torch.from_numpy(np.array([sctx])).to(self.config.device), # 0 - socially dependent, 1 - independent, 2 - same person
        }

        return sample
    

def create_datasets(config, train_name="", valid_name="", test_name=""):
    
    num_workers = 0 if config.device == "cuda" else 10
    
    pfds = TrainDS(config=config, data=load_dataset(split="train"))
    train_loader = DataLoader(pfds, batch_size=256, shuffle=True, num_workers=num_workers)

    valid_loader = DataLoader(DS(config=config, data=load_dataset(split="valid")), 
                             batch_size=256, shuffle=False, num_workers=num_workers)
    
    test_loader = DataLoader(DS(config=config, data=load_dataset(split="test")), 
                             batch_size=256, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader, None
