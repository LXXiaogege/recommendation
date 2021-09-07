"""
用户数据
"""
import torch
from torch.utils.data import Dataset
import pandas as pd


class UserDataset(Dataset):
    def __init__(self, path=r"D:\data\MIND Microsoft News Recommendation Dataset\MINDlarge_train\behaviors.tsv"):
        df = pd.read_csv(path, sep='\t', header=None)
        self.UID = df[1]
        self.history = df[3]

    def __len__(self):
        return len(self.UID)

    def __getitem__(self, item):
        return self.UID[item], self.history[item]

    def collate_fn(self, batch):
        """ history要做成一样长的 """
        uid_list, history = zip(*batch)
        history_list = []
        for idx, (history) in enumerate(batch):
            history_list.append(history)
        uid_list = torch.LongTensor(uid_list)
        history_list = torch.LongTensor(history_list)
        return uid_list, history_list


if __name__ == '__main__':
    ud = UserDataset()
    print(ud.history)
    print(type(ud.history[0]))
