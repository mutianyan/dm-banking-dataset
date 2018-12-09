import numpy as np

class splittool:
    def __init__(self,data_idx,W,K):
        self.data_idx = data_idx
        self.W = W
        self.K = K

    def rolling_window_split(self):
        train_idx = []
        test_idx = []
        loop = int((len(self.data_idx) - self.W) / self.K)
        for i in range(loop):
            train_idx.append(self.data_idx[i * self.K: i * self.K+self.W])
            test_idx.append(self.data_idx[self.W + i * self.K: self.W + (i + 1) * self.K])
        return train_idx, test_idx

