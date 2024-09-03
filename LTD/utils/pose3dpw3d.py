import pickle as pkl
from os import walk
from torch.utils.data import Dataset
import numpy as np
from utils import data_utils

from matplotlib import pyplot as plt


class Pose3dPW3D(Dataset):

    def __init__(self, path_to_data, input_n=20, output_n=10, dct_n=15, split=0):
        """

        :param path_to_data:
        :param input_n:
        :param output_n:
        :param dct_n:
        :param split:
        """
        self.path_to_data = path_to_data
        self.split = split
        self.dct_n = dct_n

        # since baselines (http://arxiv.org/abs/1805.00655.pdf and https://arxiv.org/pdf/1705.02445.pdf)
        # use observed 50 frames but our method use 10 past frames in order to make sure all methods are evaluated
        # on same sequences, we first crop the sequence with 50 past frames and then use the last 10 frame as input
        #if split == 1:
        #    their_input_n = 50
        #else:
        their_input_n = input_n
        seq_len = their_input_n + output_n

        if split == 0:
            self.data_path = path_to_data + '/train/'
        elif split == 1:
            self.data_path = path_to_data + '/test/'
        elif split == 2:
            self.data_path = path_to_data + '/validation/'
        all_seqs = []
        files = []
        for (dirpath, dirnames, filenames) in walk(self.data_path):
            files.extend(filenames)
        for f in files:
            with open(self.data_path + f, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
                joint_pos = data['jointPositions'][::2] # frequency = 2

                for i in range(len(joint_pos)):
                    seqs = joint_pos[i]
                    seqs = seqs - seqs[:, 0:3].repeat(24, axis=0).reshape(-1, 72)
                    n_frames = seqs.shape[0]
                    fs = np.arange(0, n_frames - seq_len + 1)
                    fs_sel = fs
                    for j in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + j + 1))
                    fs_sel = fs_sel.transpose()
                    seq_sel = seqs[fs_sel, :]
                    if len(all_seqs) == 0:
                        all_seqs = seq_sel
                    else:
                        all_seqs = np.concatenate((all_seqs, seq_sel), axis=0)
        
        #print(all_seqs.shape, "\n\n")
        #"2"+2
        self.all_seqs = all_seqs[:, (their_input_n - input_n):, :]

        self.dim_used = np.array(range(3, all_seqs.shape[2]))
        #all_seqs = all_seqs[:, (their_input_n - input_n):, 3:]
        n, seq_len, dim_len = all_seqs.shape
        

        #print("\n\n", all_seqs.shape, "\n\n")
        SOMOF_JOINTS = [1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]
        all_seqs = all_seqs.reshape(n, seq_len, dim_len//3, 3)[:, :, SOMOF_JOINTS].reshape(n, seq_len, len(SOMOF_JOINTS)*3)
        #print("\n\n", all_seqs.shape, "\n\n")
        n, seq_len, dim_len = all_seqs.shape
        self.all_seqs = all_seqs
        self.dim_used = np.array(range(3*len(SOMOF_JOINTS)))
        #np.save("tmp", all_seqs[:100])
        
        if split == 0:
            x_amass = np.load("../data/amass/amass-cmu_blm_troje.npy", allow_pickle=True).reshape(-1, 30, 39)
            self.all_seqs = np.concatenate((self.all_seqs, x_amass), axis=0)
            all_seqs = self.all_seqs

        
        if split == 1:
            import json
            with open("../data/somof/3dpw_test_inout.json") as json_file:
                test_gt = np.array(json.load(json_file))
                all_seqs = test_gt.reshape(test_gt.shape[0]*2, 30, 39)
            self.all_seqs = all_seqs

        print("\n\n", split, all_seqs.shape, "\n\n")   
         
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
