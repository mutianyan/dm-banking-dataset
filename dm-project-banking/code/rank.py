import pandas as pd
import numpy as np
from sklearn import svm
from split import rolling_window_split



# experiment adding and deleting columns' effect 

# get the order of insertion
rank_list = [
'duration', 'poutcome', 'month', 'pdays', 'previous', 'age', 
'contact', 'housing', 'job', 'day', 'campaign', 'education', 'balance', 'loan', 'marital', 'default']

# smaple around 5000 (make it balanced?) from the all_after_discretion_of_continuous_val.csv

data_set =pd.read_csv('../data/all_after_discretion_of_continuous_val.csv')


data_suffle = data_set.sample(frac=0.25)
train_ratio = 0.638
train_idx = int(train_ratio * data_suffle.shape[0])

train_data = data_suffle[0:train_idx]
test_data = data_suffle[train_idx+1:-1]
dataframe_train = pd.DataFrame(train_data)
dataframe_train.to_csv("../data/sampled_after_discr_train.csv")
dataframe_test = pd.DataFrame(test_data)
dataframe_test.to_csv("../data/smapled_after_discr_test.csv")

print('train size: ', dataframe_train.shape, 'test size: ', dataframe_test.shape)

# apply svm to the sample dataset

# keep only if the insertion of a column improve the accuracy

if False:
	# without training beforehand until training for feeding each invidual window
	print("using sliding window")
	array = df.values
	train_indices, test_indices =rolling_window_split(array, 10000,10)
	total_score = 0
	for train_index, test_index in zip(train_indices, test_indices):
			train_data=array.take(train_index, axis=0)
			test_data=array.take(test_index, axis=0)
			train_y =  train_data[:, 8] # 10
			train_X = np.delete(train_data, 8, 1) #10

			test_y =  test_data[:, 8] # 10
			test_X = np.delete(test_data, 8, 1) # 10

			#model = svm.SVC(kernel='linear')
			model = svm.LinearSVC() # svm.LinearSVR() does not quiet well
			model.fit(train_X, train_y) # fit the model according to the given training data
			model.predict(test_X) # predict class labels for samples in test_X
			score = model.score(test_X, test_y) # return the mean accuracy on the given test data and labels
			total_score += score
	mean_score = total_score / len(test_indices)
	print(mean_score)
else:
	print("not using sliding window")
	prev_accu = 0

	df_train = pd.read_csv('../data/sampled_after_discr_train.csv')
	df_test = pd.read_csv('../data/smapled_after_discr_test.csv')

	df_train_starter = df_train[rank_list[0]]
	df_test_starter = df_test[rank_list[0]]

	for col in rank_list[1:]:
		df_train_new = df_train[col] 
		new_train= df_train_new.append(df_train_starter, ignore_index = True)
		train_y = df_train.y
		#df_train.drop('y', inplace = True, axis= 1)
		train_X = new_train

		test_y = df_test.y
		#df_test.drop('y', inplace = True, axis = 1)
		df_test_new = df_test[col] 
		new_test= df_test_new.append(df_test_starter, ignore_index = True)
		test_X = new_test

		# model = svm.SVC(kernel='linear') # time complexity is more than quadratic with more than a couple of 10000 samples.
		model = svm.LinearSVC() # more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples
		model.fit(train_X, train_y)
		y_gnb = model.predict(test_X)
		score = model.score(test_X, test_y)
		print(list(df_train),' : ', score)

		if score <= prev_accu:
			new_train.drop(df_train_new)
			continue
		prev_accu = score






