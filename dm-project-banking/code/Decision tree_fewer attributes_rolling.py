import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


names = ['age', 'job', 'balance', 'month', 'duration', 'pdays',	'previous',	'poutcome',	'y']
data = pd.read_csv('../data/age_job_month_previous_poutcome_balance_duration_pdays.csv',
                   header=None, index_col=False, names=names, low_memory=False)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1:].values


def rolling_window_split(data, W, K):
    loop = int((data.shape[0]-W)/K)
    for i in range(loop):
        yield np.arange(i*K, i*K+W), np.arange(W+i*K, W+(i+1)*K)


clf = tree.DecisionTreeClassifier(max_depth=5)
for train_index, test_index in rolling_window_split(X, 10000, 10):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)


predicted = clf.predict(X)
cnf_matrix = confusion_matrix(y, predicted)
print(' Performance on test Set')
print('%2s%10d%10d' % (' ', 0, 1))
print('-' * 24)
print('%2d%10d%10d' % (0, cnf_matrix[0][0], cnf_matrix[0][1]))
print('%2d%10d%10d' % (1, cnf_matrix[1][0], cnf_matrix[1][1]))
print(classification_report(y, predicted, target_names=['0', '1']))
print('Accuracy: {}'.format(accuracy_score(y, predicted)))
fpr, tpr, thresholds = roc_curve(y, predicted, pos_label=1)
roc_auc = auc(fpr, tpr)
print('ROC AUC: {}'.format(roc_auc))

lw = 1
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC and ROC')
plt.legend(loc="lower right")
plt.figure()


from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=names[:-1], class_names=['success', 'failure'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("mytree.pdf")

precision, recall, _ = precision_recall_curve(y, predicted)
average_precision = average_precision_score(y, predicted)
plt.plot(recall, precision, color='darkorange', lw=lw,
         label='Precision-recall for y (area = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')
plt.legend(loc="lower right")
plt.show()

