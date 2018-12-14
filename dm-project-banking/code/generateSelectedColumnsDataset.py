import csv
import pandas as pd
from sklearn import svm
#from svm import runSVM
# plot
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def splitTrainAndTest(filename ,frac = 1):
	data_set = pd.read_csv('../data/%s.csv' % filename)
	data_suffle = data_set.sample(frac=1)
	train_ratio = 0.638
	train_idx = int(train_ratio * data_suffle.shape[0])

	train_data = data_suffle[0:train_idx]
	test_data = data_suffle[train_idx+1:-1]
	dataframe_train = pd.DataFrame(train_data)
	dataframe_train.to_csv("../data/%s_train.csv" % filename, index = False)
	dataframe_test = pd.DataFrame(test_data)
	dataframe_test.to_csv("../data/%s_test.csv" % filename, index = False)





df = pd.read_csv('../data/all_after_discretion_of_continuous_val.csv')
 
# select columns based on rule 1 ['duration', 'balance', 'pdays', 'poutcome', 'contact', 'housing', 'job', 'campaign', 'loan', 'marital'] 

# select columns based on rule 2 ['duration', 'poutcome', 'contact', 'housing', 'day', 'loan']

# select columns based on rule 3 ['duration', 'balance', 'pdays', 'poutcome', 'contact', 'housing', 'job', 'campaign', 'loan', 'marital'] 

# mannually added y columns for the selection

"""
# already geneated
df1 = df[['duration', 'balance', 'pdays', 'poutcome', 'contact', 'housing', 'job', 'campaign', 'loan', 'marital', 'y']]
df1.to_csv('../data/all_rule1.csv', index = False)

df2 = df[['duration', 'poutcome', 'contact', 'housing', 'day', 'loan', 'y']]
df2.to_csv('../data/all_rule2.csv', index = False)

df3 = df[['duration', 'balance', 'pdays', 'poutcome', 'contact', 'housing', 'job', 'campaign', 'loan', 'marital', 'y']]
df3.to_csv('../data/all_rule3.csv', index = False)

# best so far for SVM
df4 = df[['month', 'duration', 'pdays', 'age', 'contact', 'housing', 'job', 'education', 'y']]
df4.to_csv('../data/all_rule4.csv', index = False)
"""

# best so far for Random Forest
df5 = df[['month', 'duration', 'pdays', 'poutcome', 'contact', 'default', 'y']]
df5.to_csv('../data/all_rule5.csv', index = False)
"""



# not used
#splitTrainAndTest('all_rule1')
#splitTrainAndTest('all_rule2')
#splitTrainAndTest('all_rule3')
#splitTrainAndTest('expanded_all_rule4') # expanded and discritezted


""" 
splitTrainAndTest('all_rule5')
# already generated
#splitTrainAndTest('all_rule4') # best so far
#splitTrainAndTest('all_after_expand_and_discretion')
#splitTrainAndTest('all_after_discretion_of_continuous_val')


def drawConfusionMatrix(y_gnb, test_y):
    skplt.metrics.plot_confusion_matrix(y_gnb, test_y, normalize=True)
    plt.show()


def calAUC(y_true, y_score):
	return roc_auc_score(y_true, y_score)

# general function
def runSVM(filename):
	print("all_after_discretion_of_continuous_val not using sliding window")
	df_train = pd.read_csv('../data/%s_train.csv'% filename)
	df_test = pd.read_csv('../data/%s_test.csv'% filename)

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
	print('%s : ' % filename, score)
	print('auc socre :', calAUC(test_y, score))
	drawConfusionMatrix(y_gnb, test_y)
	plt.savefig(filename+'.png')

def runLR(filename):
	df_train = pd.read_csv('../data/%s_train.csv' % filename)
	df_test = pd.read_csv('../data/%s_test.csv' % filename)

	train_y = df_train.y
	df_train.drop('y', inplace=True, axis=1)
	train_X = df_train

	test_y = df_test.y
	df_test.drop('y', inplace=True, axis=1)
	test_X = df_test

	#model
	clf = LogisticRegression(penalty='l1', C=10000, class_weight='balanced', solver='saga', warm_start=True, n_jobs=3)

	clf.fit(train_X, train_y)
	pred_y = clf.predict(test_X)
	prob = clf.predict_prob(test_X, test_y)
	print('auc socre :', calAUC(test_y, prob))
	drawConfusionMatrix(pred_y,test_y)

def runRandomForest(filename):
	df_train = pd.read_csv('../data/%s_train.csv' % filename)
	df_test = pd.read_csv('../data/%s_test.csv' % filename)

	train_y = df_train.y
	df_train.drop('y', inplace=True, axis=1)
	train_X = df_train

	test_y = df_test.y
	df_test.drop('y', inplace=True, axis=1)
	test_X = df_test

	#model
	max_acc =0
	#for i in range(10,100,10):
	# n_estimators=, max_depth=3, max_features=8,min_samples_split=10, bootstrap=True,class_weight='balanced_subsample',n_jobs=3
	forest = RandomForestClassifier(n_estimators= 20, max_depth=3,min_samples_split=10, bootstrap=True,class_weight='balanced_subsample',n_jobs=3)
	forest.fit(train_X,train_y)
	pred_y = forest.predict(test_X)
	score = forest.score(test_X,test_y)
	drawConfusionMatrix(pred_y,test_y)


for filename in ['all_after_discretion_of_continuous_val', 'all_after_expand_and_discretion', 'all_rule5']:
	#runSVM(filename)
	print(filename)
	runLR(filename)
	#runRandomForest(filename)


# 'all_rule4', 'expanded_all_rule4' are for svm





