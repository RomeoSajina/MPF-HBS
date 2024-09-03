import torch.utils.data as data
import torch
import numpy as np
from fncs import *


class DATA(data.Dataset):
    def __init__(self, dataset='3dpw'):
        
        #self.data=np.load('./mocap/train_3_120_mocap.npy',allow_pickle=True)
        
        if dataset == "3dpw":

            self.data = load_original_3dw(input_window=16, output_window=14, split="train", frequency=2)

            self.data = self.data.reshape(-1, 2, 30, 39)

            x_amass = np.load("../data/amass/amass-cmu_blm_troje.npy", allow_pickle=True).reshape(-1, 1, 30, 39)
            #x_amass = np.concatenate((x_amass, x_amass[np.random.permutation(len(x_amass))]), axis=1)
            x_amass = np.concatenate((x_amass[:len(x_amass)//2], x_amass[len(x_amass)//2:len(x_amass)//2*2]), axis=1)

            self.data = np.concatenate((self.data, x_amass), axis=0)
        
        elif dataset == "handball_shot":
            self.data = load_dataset(split="train").reshape(-1, 2, 30, 39)


        print(dataset, "train data:", self.data.shape, "\n\n")
        
        self.len=len(self.data)
        

            
    def __getitem__(self, index):
        
        input_seq=self.data[index][:,:16,:]#[:,::2,:]#input, 30 fps to 15 fps
        output_seq=self.data[index][:,16:,:]#[:,::2,:]#output, 30 fps to 15 fps
        last_input=input_seq[:,-1:,:]
        output_seq=np.concatenate([last_input,output_seq],axis=1)

        return input_seq,output_seq
        
        
        
    def __len__(self):
        return self.len



class TESTDATA(data.Dataset):
    def __init__(self, dataset='3dpw'):
        
        #self.data=np.load('./mocap/test_3_120_mocap.npy',allow_pickle=True)
        
        if dataset == "3dpw":
            import json
            with open("../data/somof/3dpw_test_inout.json") as json_file:
                self.data = np.array(json.load(json_file))
        
        elif dataset == "handball_shot":
            self.data = load_dataset(split="test")
            
        self.data = self.data.reshape(-1, 2, 30, 39)
        
        self.len=len(self.data)
        
        print(dataset, "test data:", self.data.shape, "\n\n")
        

    def __getitem__(self, index):

        input_seq=self.data[index][:,:16,:]#[:,::2,:]#input, 30 fps to 15 fps
        output_seq=self.data[index][:,16:,:]#[:,::2,:]#output, 30 fps to 15 fps
        last_input=input_seq[:,-1:,:]
        output_seq=np.concatenate([last_input,output_seq],axis=1)

        return input_seq,output_seq

    def __len__(self):
        return self.len
