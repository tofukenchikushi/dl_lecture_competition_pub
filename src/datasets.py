import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
import torchaudio.transforms as transforms

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = self.load_data(os.path.join(data_dir, f"{split}_X"))
        self.subject_idxs = self.load_data(os.path.join(data_dir, f"{split}_subject_idxs"))
        
        if split in ["train", "val"]:
            self.y = self.load_data(os.path.join(data_dir, f"{split}_Y"))
            assert len(torch.unique(torch.tensor(self.y))) == self.num_classes, "Number of classes do not match."
        
        # MFCC変換器の初期化
        self.mfcc_transform = transforms.MFCC(
            sample_rate=200,  # サンプリングレートを指定
            n_mfcc=13,  # MFCCの次元数
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
        )

    def load_data(self, dir_path: str) -> torch.Tensor:
        files = sorted(os.listdir(dir_path))
        data = [np.load(os.path.join(dir_path, file)) for file in files]
        return torch.tensor(np.concatenate(data, axis=0))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        X_mfcc = self.mfcc_transform(X)  # MFCC特徴量の計算
        if hasattr(self, "y"):
            return X_mfcc, self.y[i], self.subject_idxs[i]
        else:
            return X_mfcc, self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return 13  # MFCCの次元数に合わせる
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2] // 160  # hop_lengthに合わせたシーケンス長