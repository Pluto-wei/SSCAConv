import torch
import numpy as np
from scipy import io as sio
from SSCANet import SSCANet as SSCANet
import h5py

device=torch.device('cuda:0')

def load_set(file_path):
    data = h5py.File(file_path)  # 8,64,64,1258

    # tensor type:
    lms = torch.from_numpy(np.array(data.get('lms')) / 2047.0) # CxHxW = 8x256x256
    pan = torch.from_numpy(np.array(data.get('pan')) / 2047.0)   # HxW = 256x256
    print(lms.shape,pan.shape)

    # data = sio.loadmat(file_path)  # HxWxC=256x256x8
    #
    # # tensor type:
    # lms = torch.from_numpy(data['lms'] / 2047.0).permute(0,3,1,2)  # CxHxW = 8x256x256
    # pan = torch.from_numpy((data['pan'] / 2047.0)).unsqueeze(1)   # HxW = 256x256
    # print(lms.shape)

    return lms, pan


file_path = 'h5/WV3/reduce_examples/test_wv3_multiExm1.h5'


test_lms, test_pan = load_set(file_path)
test_lms = test_lms.to(device).float()

test_pan = test_pan.to(device).float()  # convert to tensor type: 1x1xHxW


model=SSCANet(32,5,True).to(device)
model.load_state_dict(torch.load('weights/wv3/1cNet_1500.pth'))
model.eval()
with torch.no_grad():
    for i in range(20):

        outputi = model(test_pan[i].unsqueeze(0),test_lms[i].unsqueeze(0))
        print(i+1)
        sri = torch.squeeze(outputi).permute(1, 2, 0).cpu().detach().numpy()
        sri = sri * 2047.0
        sio.savemat('results/WV3/reduced/output_WV3_re{}.mat'.format(i),
                    {'test_result': sri})