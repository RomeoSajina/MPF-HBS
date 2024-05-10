from dataset import load_test
import numpy as np
import argparse

def VIM(pred, GT, calc_per_frame=True, return_last=True):
    if calc_per_frame:
        pred = pred.reshape(-1, 39)
        GT = GT.reshape(-1, 39)
    errorPose = np.power(GT - pred, 2)
    errorPose = np.sum(errorPose, 1)
    errorPose = np.sqrt(errorPose)
    
    if return_last:
        errorPose = errorPose[-1]
    return errorPose


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="3dpw", help='dataset name')
args = parser.parse_args()

test = load_test(args.dataset)

y_pred = test[..., 15:16, :, :].repeat(14, axis=-3)
y_test = test[..., -14:, :, :]

vims = " ".join( [ str(round(np.mean( [(VIM(pred[0][:LEN], gt[0][:LEN]) + VIM(pred[1][:LEN], gt[1][:LEN])) / 2. for pred, gt in zip(y_pred, y_test)] ) * 100, 1)) for LEN in [2, 4, 8, 10, 14]]  )

print("Test [100ms 240ms 500ms 640ms 900ms]:", vims)
np.save("../data/predictions/{0}_zero_velocity".format(args.dataset), y_pred)

