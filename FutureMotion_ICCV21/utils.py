import os
import pickle
import numpy as np
import torch


def load_data_3dpw_multiperson(dataset_dir="../data/3dpw/", split="train"):
    # TRAIN AND TEST SETS ARE REVERSED FOR SOMOF
    SPLIT_3DPW = {
        "train": "test",
        "val": "validation",
        "valid": "validation",
        "test": "train"
    }
    datalist = []
    
    path_to_data = os.path.join(dataset_dir, "sequenceFiles", SPLIT_3DPW[split])
    
    for pkl in os.listdir(path_to_data):
        with open(os.path.join(path_to_data, pkl), 'rb') as reader:
            annotations = pickle.load(reader, encoding='latin1')

        all_person_tracks = []
        
        for actor_index in range(len(annotations['genders'])):

            joints_2D = annotations['poses2d'][actor_index].transpose(0, 2, 1)
            joints_3D = annotations['jointPositions'][actor_index]
            
            track_joints = []
            track_mask = []

            for image_index in range(len(joints_2D)): # range(t1, t2):
                path =  os.path.join(dataset_dir, "imageFiles", os.path.splitext(pkl)[0], f"image_{str(image_index).zfill(5)}.jpg")
                J_3D_real = joints_3D[image_index].reshape(-1, 3)
                J_3D_mask = np.ones(J_3D_real.shape[:-1])
                track_joints.append(J_3D_real)
                track_mask.append(J_3D_mask)

            all_person_tracks.append((np.asarray(track_joints), np.asarray(track_mask)))

        datalist.append(all_person_tracks)

    return datalist


def to_seq_length(seq_list, SL, freq):

    seqs = []
    for orig_seq in [x for x in seq_list if x.shape[1] >= SL*freq]:

        for sampled_s in [orig_seq[:, i::freq] for i in range(freq)]:

            seqs.extend( [sampled_s[:, i:i+SL] for i in range(sampled_s.shape[1]-SL)] ) 

    return np.array(seqs)


def load_original_3dw(input_window=16, output_window=14, split="train", frequency=2):
    SL = input_window + output_window
    freq = frequency
    SOMOF_JOINTS = [1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]
    
    ds = load_data_3dpw_multiperson(split=split)
    
    one_p_3dpw = [np.expand_dims(x[0][0], axis=0) for x in ds if len(x) == 1]
    two_p_3dpw = [np.concatenate([np.expand_dims(x[0][0], axis=0), np.expand_dims(x[1][0], axis=0)], axis=0) for x in ds if len(x) == 2]

    one_p_3dpw = to_seq_length(one_p_3dpw, SL, freq)[..., SOMOF_JOINTS, :]
    two_p_3dpw = to_seq_length(two_p_3dpw, SL, freq)[..., SOMOF_JOINTS, :]
    
    return two_p_3dpw[:, :, :input_window], two_p_3dpw[:, :, input_window:], one_p_3dpw[:, :, :input_window], one_p_3dpw[:, :, input_window:]


def keypoint_mpjpe(pred, gt):
    error = np.linalg.norm(pred - gt, ord=2, axis=-1).mean()
    return error

def calc_kp_mpjpe(model, data, SIZE=14):
    errs = []
    for i, vx in enumerate(data):

        voutputs = model(vx, False)

        er0 = keypoint_mpjpe(voutputs["z0"][:, :SIZE].reshape(*vx["out_keypoints0"][:, :SIZE].shape).float().detach().cpu().numpy(), vx["out_keypoints0"][:, :SIZE].float().detach().cpu().numpy()) 
        er1 = keypoint_mpjpe(voutputs["z1"][:, :SIZE].reshape(*vx["out_keypoints1"][:, :SIZE].shape).float().detach().cpu().numpy(), vx["out_keypoints1"][:, :SIZE].float().detach().cpu().numpy())

        errs.append(er0)
        errs.append(er1)

    return np.mean(errs)


def VIM(pred, GT, calc_per_frame=True, return_last=True):
    if calc_per_frame:
        pred = pred.reshape(-1, 39)
        GT = GT.reshape(-1, 39)
    #print(pred.shape, GT.shape)
    errorPose = np.power(GT - pred, 2)
    errorPose = np.sum(errorPose, 1)
    errorPose = np.sqrt(errorPose)
    
    if return_last:
        errorPose = errorPose[-1]
    return errorPose
