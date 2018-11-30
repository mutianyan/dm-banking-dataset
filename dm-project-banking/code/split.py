import numpy as np


def rolling_window_split(data,W,K):
	train_idx=[]
	test_idx=[]
	loop=int((data.shape[0]-W)/K)
	for i in range(loop):
		train_idx.append(range(i*K,i*K+W))
		test_idx.append(range(W+i*K,W+(i+1)*K))
	return train_idx,test_idx


