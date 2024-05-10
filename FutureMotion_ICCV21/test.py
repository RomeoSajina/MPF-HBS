import argparse
import torch
from model import Futuremotion_ICCV21
import numpy as np
from utils import VIM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Futuremotion_ICCV21().to(device)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="3dpw", help='dataset name')
parser.add_argument('--ckp', type=str)
args = parser.parse_args()


if args.dataset == "3dpw":
    from dataset_3dpw import create_datasets
else:
    from dataset_handball_shot import create_datasets

_, _, test = create_datasets()


model.load_state_dict(torch.load(args.ckp))
model.eval()


y_pred = []
y_test = []
for i, inp in enumerate(test):
    with torch.no_grad():

        out = model(inp, False)

    p0 = out["z0"][:, :14].reshape(1, 14, 13, 3).float().detach().cpu().numpy()
    p1 = out["z1"][:, :14].reshape(1, 14, 13, 3).float().detach().cpu().numpy()

    del out["z0"], out["z1"]

    y_pred.append(np.concatenate((p0, p1), axis=0))
    y_test.append(np.concatenate((inp["out_keypoints0"].float().detach().cpu().numpy(), inp["out_keypoints1"].float().detach().cpu().numpy()), axis=0))
    print_wei = False

y_pred = np.array(y_pred)
vims = " ".join( [ str(round(np.mean( [(VIM(pred[0][:LEN], gt[0][:LEN]) + VIM(pred[1][:LEN], gt[1][:LEN])) / 2. for pred, gt in zip(y_pred, y_test)] ) * 100, 1)) for LEN in [2, 4, 8, 10, 14]]  )

print("Test [100ms 240ms 500ms 640ms 900ms]:", vims)
np.save("../data/predictions/{0}_future_motion_iccv21".format(args.dataset), y_pred)

