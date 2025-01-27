from sklearn.linear_model import LogisticRegression
import pandas as pd
from splittool import *
from sklearn import svm
# from precess2limitedattr import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import scikitplot as skplt
import numpy as np
from sklearn.metrics import accuracy_score
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

keylist =['age','job','month','previous','poutcome', 'balance','duration', 'pdays']




data_set = pd.read_csv('../data/all_after_discretion_of_continuous_val.csv',delimiter = ',')
# data_set = precess2limitedattr(keylist)
# data_set, label = data_set.getdata()

print("precess data finished!")
W = 10000
K = 10
test_num = 1200

data_eval_idx = list(range(len(data_set)-W-test_num,len(data_set)))
eval_train_idx,eval_test_idx = splittool(data_eval_idx,W,K).rolling_window_split()

label=data_set['y']
data_set = data_set.drop('y',axis=1)

train_X = data_set[0:len(data_set)-test_num]
train_y = label[0:len(data_set)-test_num]

# def shuffleIdx(idxlist):
#     return random.sample(idxlist,len(idxlist))


# idx = shuffleIdx(list(range(0,len(data_set)-test_num)))

# clf = LogisticRegression()
# clf = svm.LinearSVC()
# clf = RandomForestClassifier()
clf = tree.DecisionTreeClassifier()
# idx_test = shuffleIdx(list(range(len(train_X)-test_num-1,len(train_X))))
clf.fit(data_set.values,label)

print("no window")
print(clf.score(data_set.values[-test_num:],label[-test_num:]))

test_X = data_set[len(data_set)-test_num:len(data_set)]
test_y = label[len(data_set)-test_num:len(data_set)]
# pred_y = clf.predict_proba(test_X)
# score = roc_auc_score(test_y, pred_y[:,1])
# print(score)

cf_matrix=[]
acc=[]

count =0
for i in range(len(eval_test_idx)):
    eval_train_y = label[eval_train_idx[i][0]:eval_train_idx[i][0]+len(eval_train_idx[i])]
    eval_train_X = data_set[eval_train_idx[i][0]:eval_train_idx[i][0]+len(eval_train_idx[i])]
    # train_idx =  shuffleIdx(list(range(len(eval_train_X))))
    # eval_tr_X_sff = eval_train_X[train_idx]
    # eval_tr_y_sff = eval_train_y[train_idx]

    eval_test_X = data_set[eval_test_idx[i][0]:eval_test_idx[i][0]+len(eval_test_idx[i])]
    eval_test_y = label[eval_test_idx[i][0]:eval_test_idx[i][0]+len(eval_test_idx[i])]

    # test_idx = shuffleIdx(list(range(len(test))))

    clf.fit(eval_train_X,eval_train_y)
    # yhat_prob =clf.predict_proba(eval_test_X)
    # yhat = clf.predict(eval_test_X)
    score = clf.score(eval_test_X,eval_test_y)
    # acc.append(accuracy_score(eval_test_y,yhat))
    if sum(eval_test_y)==K or sum(eval_test_y)==0:
        count+=1
        continue
    cf_matrix.append(score)


print(count)
print(np.mean(cf_matrix))
# print(cf_matrix)
# print(np.mean(acc))
