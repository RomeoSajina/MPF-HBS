import numpy as np
import torch
import torch.utils.data as data
from fncs import *

class Data(data.Dataset):
    def __init__(self, dataset, split="train", scene=None, device='cuda', input_time=25, output_time=25):

        if dataset == "3dpw":
            if split == "train":
                self.data = load_original_3dw(input_window=16, output_window=14, split="train", frequency=2)

                self.data = self.data.reshape(-1, 2, 30, 13, 3)

                x_amass = np.load("../data/amass/amass-cmu_blm_troje.npy", allow_pickle=True).reshape(-1, 1, 30, 13, 3)
                x_amass = np.concatenate((x_amass[:len(x_amass)//2], x_amass[len(x_amass)//2:len(x_amass)//2*2]), axis=1)
                self.data = np.concatenate((self.data, x_amass), axis=0)

            elif split == "test":
                import json
                with open("../data/somof/3dpw_test_inout.json") as json_file:
                    self.data = np.array(json.load(json_file))
                self.data = self.data.reshape(-1, 2, 30, 13, 3)
                
        elif dataset == "handball_shot":
            self.data = load_dataset(split=split)
            print(dataset, split, "loaded:", self.data.shape)

        self.len = len(self.data)
        self.device = device
        self.dataset = dataset
        self.input_time = input_time
        self.output_time = output_time


    def __getitem__(self, index):
        data = self.data[index]
        input_seq = data[:, :self.input_time, ...]
        output_seq = data[:, self.input_time:,...]
        input_seq = torch.as_tensor(input_seq, dtype=torch.float32).to(self.device)
        output_seq = torch.as_tensor(output_seq, dtype=torch.float32).to(self.device)
        
        # last_input = input_seq[:, -1:, :]
        # output_seq = torch.cat([last_input, output_seq], dim=1)
        # input_seq = input_seq.reshape(input_seq.shape[0], input_seq.shape[1], -1)
        # output_seq = output_seq.reshape(output_seq.shape[0], output_seq.shape[1], -1)

        return input_seq, output_seq

    def __len__(self):
        return self.len




