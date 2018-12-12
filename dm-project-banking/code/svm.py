import pandas as pd
import numpy as np
from sklearn import svm

# Packages for visuals
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

# Pickle package
import pickle
from split import rolling_window_split


def drawConfusionMatrix(y_gnb, test_y):
    skplt.metrics.plot_confusion_matrix(y_gnb, test_y, normalize=True)
    plt.show()



def drawPRGraph(train_X, train_y, test_X, test_y, probas):
    skplt.metrics.plot_precision_recall(test_y, probas)
    plt.show()


def drawRoc(test_y, probas):
    skplt.metrics.plot_roc(test_y, probas)
    plt.show()


def calScores(test_y, y_gnb, probas):
    precision, recall, thresholds = skplt.metrics.precision_recall_curve(test_y, probas)
    # calculate F1 score
    f1 = f1_score(test_y, y_gnb)
    # calculate precision-recall AUC
    aucc = auc(recall, precision) # accuracy 1
    # calculate average precision score
    ap = average_precision_score(test_y, probas)
    print('f1=%.3f auc=%.3f ap=%.3f' % (f1, aucc, ap))



#df = pd.read_csv('../data/afterTransform.csv')
df = pd.read_csv('../data/age_job_month_previous_poutcome_balance_duration_pdays.csv')
#using sliding window
# using 

if True:
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
	df_train = pd.read_csv('../data/afterTransform_train.csv')
	df_test = pd.read_csv('../data/afterTransform_test.csv')

	train_y = df_train.y
	df_train.drop('y', inplace = True, axis= 1)
	train_X = df_train

	test_y = df_test.y
	df_test.drop('y', inplace = True, axis = 1)
	test_X = df_test

	# model = svm.SVC(kernel='linear') # time complexity is more than quadratic with more than a couple of 10000 samples.
	model = svm.LinearSVC() # more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples
	model.fit(train_X, train_y)
	y_gnb = model.predict(test_X)
	score = model.score(test_X, test_y)
	print(score)
	# adjust the values of the parameters (combination of (C, gamma), kernel options, decision function shape)
	# print the hyperplance (coefficient, intercept,)

# drawing graphs
drawConfusionMatrix(y_gnb, test_y)
#drawPRGraph(train_X, train_y, test_X, test_y, probas)
#drawRoc(test_y, probas)

# calculate statistics
#calScores(test_y, y_gnb, probas)



"""
	# random sampling
	data_suffle = df.sample(frac=1)
	train_ratio = 0.638
	train_idx = int(train_ratio * data_suffle.shape[0])

	train_data = data_suffle[0:train_idx]
	test_data = data_suffle[train_idx+1:-1]
	dataframe_train = pd.DataFrame(train_data)
	dataframe_train.to_csv("../data/afterTransform_train.csv")
	dataframe_test = pd.DataFrame(test_data)
	dataframe_test.to_csv("../data/afterTransform_test.csv")
	
"""


"""
train_X = [[0, 0], [1, 1]]
train_y = [0, 1]

test_X = [[0, 0], [1, 1]]
test_y = [1,0]

# prepare the data
#sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70});

model = svm.SVC(kernel='linear')
model.fit(train_X, train_y)
model.predict(test_X)
score = model.score(test_X, test_y)

print(score)
"""
"""
drawConfusionMatrix(y_gnb, test_y)
drawPRGraph(train_X, train_y, test_X, test_y, probas)
drawRoc(test_y, probas)
"""

# draw precision- recall curve



"""
			# get support vectors
			print(model.support_vectors_)
			# get indices of support vectors
			print(model.support_) 
			# get number of support vectors for each class
			print(model.n_support_) 
"""