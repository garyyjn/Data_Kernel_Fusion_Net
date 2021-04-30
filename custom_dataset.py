import numpy as np
import csv
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import skvideo.io
import os

class ucf_c3d(Dataset):#batch, 3, 16, 112, 112
    def __init__(self, annotations, video_dir, kernels = ['original'], transform=None):
        self.action_annotations = pd.read_csv(annotations, index_col = 0)
        self.video_dir = video_dir
        self.kernels = kernels
        self.transform = transform

    def __len__(self):
        return len(self.action_annotations)

    def __getitem__(self, idx):#todo preprocess videos, currently too slow
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item_row = self.action_annotations.iloc[idx]
        video_name = item_row['video_name']

        action = item_row['action']
        sample = {'action':action}
        try:
            for kernel in self.kernels:
                if kernel == 'original':
                    video_matrix = torch.tensor(np.load(os.path.join(self.video_dir,'original', video_name)).astype(np.float32))
                    sample['original'] = video_matrix
                if kernel == 'four_frames':
                    video_matrix = torch.tensor(np.load(os.path.join(self.video_dir,'original', video_name)).astype(np.float32))
                    sample['four_frames'] = video_matrix
                if kernel == 'diff':
                    video_matrix = torch.tensor(np.load(os.path.join(self.video_dir,'original', video_name)).astype(np.float32))
                    sample['diff'] = video_matrix
        except():
            print("loading error")

        return sample

    def video2matrix(self, path):  # returns shape: frames x height x width x color_dims
        videodata = skvideo.io.vread(path)
        return np.array(videodata)


