import numpy as np


def rolling_window_split(data,W,K):
	train_idx=[]
	test_idx=[]
	loop=int((data.shape[0]-W)/K)
	for i in range(loop):
		train_idx.append(range(i*K,i*K+W))
		test_idx.append(range(W+i*K,W+(i+1)*K))
	return train_idx,test_idx

data = np.array([[1,2], [1,2], [1,2], [1,2], [1,2], [1,2]])

train_idx, test_idx = rolling_window_split(data, 2, 1)
print("training index : ", train_idx)
print("testing index : ", test_idx)


arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
delete_row= np.delete(arr, 1, 0)
print(delete_row)

delete_col= np.delete(arr, 1, 1) 
print(delete_col)
