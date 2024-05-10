import os
import json
import numpy as np

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
    
    return XY


def load_dataset(split="train", SL=30, freq=2):
    data = np.load("../data/handball_shot/{0}.npy".format(split))
    print("Loading {0} split of dataset:".format(split), data.shape)
    return data


def load_test(dataset):
    if dataset == "3dpw":
        return l("test")
    elif dataset == "handball_shot":
        return load_dataset(split="test")
    
    

    