import pickle as pkl
from os import walk
from torch.utils.data import Dataset
import numpy as np
from utils import data_utils

from matplotlib import pyplot as plt

import os

def load_dataset(split="train", SL=30, freq=2):
    data = np.load("../data/handball_shot/{0}.npy".format(split))
    print("Loading {0} split of dataset:".format(split), data.shape)
    return data


class DS3D(Dataset):

    def __init__(self, path_to_data, input_n=20, output_n=10, dct_n=15, split=0):

        if split == 0:
            split_name = "train"
        elif split == 1:
            split_name = "test"
        elif split == 2:
            split_name = "valid"
  
        all_seqs = load_dataset(split=split_name)
        all_seqs = all_seqs.reshape(-1, 30, 39)

        self.dim_used = np.array(range(39))
        n, seq_len, dim_len = all_seqs.shape
        
        self.all_seqs = all_seqs

        print("\n\n", split_name, all_seqs.shape, "\n\n")   
         
        all_seqs = all_seqs.transpose(0, 2, 1)
        all_seqs = all_seqs.reshape(-1, input_n + output_n)
        all_seqs = all_seqs.transpose()

        dct_m_in, _ = data_utils.get_dct_matrix(input_n + output_n)
        dct_m_out, _ = data_utils.get_dct_matrix(input_n + output_n)
        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)
        input_dct_seq = np.matmul(dct_m_in[0:dct_n, :], all_seqs[i_idx, :])
        input_dct_seq = input_dct_seq.transpose().reshape(-1, dim_len, dct_n)
        # input_dct_seq = input_dct_seq.reshape(-1, dim_len * dct_used)

        output_dct_seq = np.matmul(dct_m_out[0:dct_n, :], all_seqs)
        output_dct_seq = output_dct_seq.transpose().reshape(-1, dim_len, dct_n)
        # output_dct_seq = output_dct_seq.reshape(-1, dim_len * dct_used)

        self.input_dct_seq = input_dct_seq
        self.output_dct_seq = output_dct_seq
          
        #print("\n\n", input_dct_seq.shape, output_dct_seq.shape, "\n\n")
          
    def __len__(self):
        return np.shape(self.input_dct_seq)[0]

    def __getitem__(self, item):
        return self.input_dct_seq[item], self.output_dct_seq[item], self.all_seqs[item]
