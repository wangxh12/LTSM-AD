import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler



class MinnesotaSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.args = args
        self.root_path = root_path
        self.win_size = win_size
        self.flag = flag
        self.step = step

        self.target_fields = [
            'navalt', 'alt', 'h', 'navvn', 'navve', 'navvd',
            'vn', 've', 'vd', 'p', 'q', 'r'
        ]

        self.train_file = getattr(args, "train_file", "ThorFlight104.csv")
        self.test_file = getattr(args, "test_file", "ThorFlight121.csv")

        self.train_path = os.path.join(root_path, self.train_file)
        self.test_path = os.path.join(root_path, self.test_file)

        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        if "label" not in test_df.columns:
            raise ValueError(f"'label' column not found in {self.test_path}")

        # 保持字段顺序，不要用 set
        common_fields = [
            f for f in self.target_fields
            if f in train_df.columns and f in test_df.columns
        ]

        if len(common_fields) != len(self.target_fields):
            missing = [f for f in self.target_fields if f not in common_fields]
            raise ValueError(f"Missing target fields: {missing}")

        train_raw = train_df[common_fields].values.astype("float32")
        test_raw = test_df[common_fields].values.astype("float32")
        test_label = test_df["label"].values.astype("float32")

        train_raw = np.nan_to_num(train_raw)
        test_raw = np.nan_to_num(test_raw)

        # 从 ThorFlight104 里切 train / val
        border = int(len(train_raw) * 0.8)
        train_part = train_raw[:border]
        val_part = train_raw[border:]

        # 只用训练段 fit scaler，避免验证集泄漏
        # self.scaler = StandardScaler()
        self.scaler = MinMaxScaler()
        self.scaler.fit(train_part)

        self.train = self.scaler.transform(train_part).astype("float32")
        self.val = self.scaler.transform(val_part).astype("float32")
        self.test = self.scaler.transform(test_raw).astype("float32")
        self.test_labels = test_label

        print(f"[Minnesota] flag={flag}")
        print(f"Train: {self.train.shape}, Val: {self.val.shape}, Test: {self.test.shape}")
        print(f"Fields: {common_fields}")
        print(f"Step: {self.step}")

    def __len__(self):
        if self.flag == "train":
            data_len = len(self.train)
        elif self.flag == "val":
            data_len = len(self.val)
        elif self.flag == "test":
            data_len = len(self.test)
        else:
            data_len = len(self.test)

        return (data_len - self.win_size) // self.step + 1

    def __getitem__(self, index):
        s_begin = index * self.step
        s_end = s_begin + self.win_size

        if self.flag == "train":
            x = self.train[s_begin:s_end]
            y = np.zeros(self.win_size, dtype="float32")
        elif self.flag == "val":
            x = self.val[s_begin:s_end]
            y = np.zeros(self.win_size, dtype="float32")
        else:
            x = self.test[s_begin:s_end]
            y = self.test_labels[s_begin:s_end]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    
    