import torch.utils.data as data
import torch
import numpy as np
import copy
import open3d as o3d
from fncs import *

class Data(data.Dataset):
    def __init__(self, dataset, mode=0, device='cuda', transform=False, opt=None):
        
        if dataset == "3dpw":

            if mode != -1:
                split = "train" if mode==0 else "valid"
                self.data = load_original_3dw(input_window=16, output_window=14, split=split, frequency=2)

                x_amass = np.load("../data/amass/amass-cmu_blm_troje.npy", allow_pickle=True).reshape(-1, 1, 30, 13, 3)
                x_amass = np.concatenate((x_amass[:len(x_amass)//2], x_amass[len(x_amass)//2:len(x_amass)//2*2]), axis=1)

                self.data = np.concatenate((self.data, x_amass), axis=0)

            else:
                import json
                with open("../data/somof/3dpw_test_inout.json") as json_file:
                    self.data = np.array(json.load(json_file))
                self.data = self.data.reshape(-1, 2, 30, 39)
                
        elif dataset == "handball_shot":
            if mode != -1:
                split_name = "train" if mode==0 else "valid"
            else:
                split_name = "test"
            self.data = load_dataset(split=split_name)#.reshape(-1, 2, 30, 39)
            print(dataset, split_name, "loaded:", self.data.shape)
        
        self.len = len(self.data)
        self.device = device
        self.dataset = dataset
        self.transform = transform
        self.input_time = opt.input_time


    def __getitem__(self, index):
        data = self.data[index]
        
        if self.transform:   # radomly rotate the scene for augmentation
            idx = np.random.randint(0, 3)
            rot = [np.pi, np.pi/2, np.pi/4, np.pi*2]
            points = self.data[index].reshape(-1, 3)
            # 读取点
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # 点旋转
            pcd_EulerAngle = copy.deepcopy(pcd)
            R1 = pcd.get_rotation_matrix_from_xyz((0, rot[idx], 0))
            pcd_EulerAngle.rotate(R1)  # 不指定旋转中心
            pcd_EulerAngle.paint_uniform_color([0, 0, 1])
            data = np.asarray(pcd_EulerAngle.points).reshape(-1, self.opt.input_time+self.opt.output_time, self.opt.n_joints*3)

        input_seq = data[:, :self.input_time, ]
        output_seq = data[:, self.input_time:, :]

        input_seq = torch.as_tensor(input_seq, dtype=torch.float32).to(self.device)
        output_seq = torch.as_tensor(output_seq, dtype=torch.float32).to(self.device)
        last_input = input_seq[:, -1:, :]
        output_seq = torch.cat([last_input, output_seq], dim=1)
        input_seq = input_seq.reshape(input_seq.shape[0], input_seq.shape[1], -1)
        output_seq = output_seq.reshape(output_seq.shape[0], output_seq.shape[1], -1)

        return input_seq, output_seq

    def __len__(self):
        return self.len




