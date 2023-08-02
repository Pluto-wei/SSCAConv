import torch.utils.data as data
import torch
import h5py
import numpy as np


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()

        data = h5py.File(file_path,'r')  # NxCxHxW = 0x1x2x3

        for key in data.keys():
            print(data[key].name)
            print(data[key].shape)

        if 'gt' in data.keys():
            # tensor type:
            gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
            gt1 = np.array(gt1, dtype=np.float32) / 2047.
            self.gt = torch.from_numpy(gt1)  # NxCxHxW:
            # print(gt1.shape)

            lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
            lms1 = np.array(lms1, dtype=np.float32) / 2047.
            self.lms = torch.from_numpy(lms1)
            # print(lms1.shape)

            pan1 = data['pan'][...]  # Nx1xHxW
            pan1 = np.array(pan1, dtype=np.float32) / 2047.  # Nx1xHxW
            self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
            # print(pan1.shape)
        else:
            # / GT
            # (3136, 31, 64, 64)
            # / HSI_up
            # (3136, 31, 64, 64)
            # / LRHSI
            # (3136, 31, 16, 16)
            # / RGB
            # (3136, 3, 64, 64)
            gt1 = data["GT"][...]  # convert to np tpye for CV2.filter
            gt1 = np.array(gt1, dtype=np.float32)
            print(np.max(gt1))
            self.gt = torch.from_numpy(gt1)  # NxCxHxW:
            # print(gt1.shape)

            lms1 = data["HSI_up"][...]  # convert to np tpye for CV2.filter
            lms1 = np.array(lms1, dtype=np.float32)
            self.lms = torch.from_numpy(lms1)
            # print(lms1.shape)

            pan1 = data['RGB'][...]  # Nx1xHxW
            pan1 = np.array(pan1, dtype=np.float32)   # Nx1xHxW
            self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
            # print(pan1.shape)

    #####必要函数
    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float() # Nx1xHxW:
            #####必要函数
    def __len__(self):
        return self.gt.shape[0]