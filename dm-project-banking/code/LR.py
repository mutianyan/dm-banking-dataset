from sklearn.linear_model import LogisticRegression
import pandas as pd
from split import *
from precess2limitedattr import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import scikitplot as skplt
import numpy as np
from sklearn.metrics import accuracy_score

keylist =['age','job','month','previous','poutcome', 'balance','duration', 'pdays']




# data = pd.read_csv('../data/afterTransform.csv',delimiter = ',')
data_set = precess2limitedattr(keylist)
data_set, label = data_set.getdata()

print("precess data finished!")
W = 20000
K = 20
test_num = 1200

data_eval_idx = list(range(len(data_set)-W-test_num,len(data_set)))
eval_train_idx,eval_test_idx = split(data_eval_idx,W,K).rolling_window_split()

# label=data['y']
# data_set = data.drop('y',axis=1)

train_X = data_set[0:len(data_set)-test_num]
train_y = label[0:len(data_set)-test_num]



clf = LogisticRegression(penalty='l1',C=10000,class_weight='balanced',solver='saga',warm_start=True,n_jobs=3)

clf.fit(train_X,train_y)

test_X = data_set[len(data_set)-test_num:len(data_set)]
test_y = label[len(data_set)-test_num:len(data_set)]
pred_y = clf.predict_proba(test_X)
score = roc_auc_score(test_y, pred_y[:,1])
print(score)

cf_matrix=[]
acc=[]
count =0
for i in range(len(eval_test_idx)):
    eval_train_X = data_set[eval_train_idx[i][0]:eval_train_idx[i][0]+len(eval_train_idx[i])]
    eval_train_y = label[eval_train_idx[i][0]:eval_train_idx[i][0]+len(eval_train_idx[i])]

    eval_test_X = data_set[eval_test_idx[i][0]:eval_test_idx[i][0]+len(eval_test_idx[i])]
    eval_test_y = label[eval_test_idx[i][0]:eval_test_idx[i][0]+len(eval_test_idx[i])]
    # clf.fit(eval_train_X,eval_train_y)
    yhat_prob =clf.predict_proba(eval_test_X)
    yhat = clf.predict(eval_test_X)
    acc.append(accuracy_score(eval_test_y,yhat))
    if sum(eval_test_y)==K or sum(eval_test_y)==0:
        count+=1
        continue
    cf_matrix.append(roc_auc_score(eval_test_y, yhat_prob[:,1]))

print(count)
print(np.mean(cf_matrix))
print(cf_matrix)
print(np.mean(acc))
