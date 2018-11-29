import csv
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

"""
fp = open("../bank.csv", "r")
fp1 = open("shortfile", "w")
for i, line in enumerate(fp):
    if i <= 31:
    	fp1.write(line)
    elif i > 31:
        break
fp.close()
fp1.close()

"""

# bank.csv from May 2008 to November 2010

data_set = pd.read_csv('../data/shortfile',delimiter = ';')

def transforForamt(dataset, diction):
    for i in range(len(dataset)):
        dataset[i] = diction[dataset[i]]
    return dataset




dic={"yes":1,"no":0}
data_set['deafult'] = transforForamt(data_set['default'],dic)
data_set['housing'] = transforForamt(data_set['housing'],dic)
data_set['loan'] = transforForamt(data_set['loan'],dic)
data_set['y'] = transforForamt(data_set['y'],dic)

# categorical and continuous variables
#https://scikit-learn.org/stable/modules/preprocessing.html


# change unkown values to different labels


data_set.loc[data_set['education'] == 'unknown', 'education'] = 'unknown_education'
data_set.loc[data_set['contact'] == 'unknown', 'contact'] = 'unknown_contact'
data_set.loc[data_set['poutcome'] == 'unknown', 'poutcome'] = 'unknown_poutcome'

# pandas get dummies for categorical variables

print(list(data_set))


s = data_set.job
one_hot=pd.get_dummies(s) # return type is dataFrame
data_set = data_set.join(one_hot)
data_set = data_set.drop('job',axis=1)

s = data_set.marital
one_hot=pd.get_dummies(s) # return type is dataFrame
data_set = data_set.join(one_hot)
data_set = data_set.drop('marital',axis=1)

s = data_set.education
one_hot=pd.get_dummies(s) # return type is dataFrame
data_set = data_set.join(one_hot)
data_set = data_set.drop('education',axis=1)

s = data_set.contact
one_hot=pd.get_dummies(s) # return type is dataFrame
data_set = data_set.join(one_hot)
data_set = data_set.drop('contact',axis=1)

s = data_set.poutcome
one_hot=pd.get_dummies(s) # return type is dataFrame
data_set = data_set.join(one_hot)
data_set = data_set.drop('poutcome',axis=1)

# ?
s = data_set.month
one_hot=pd.get_dummies(s) # return type is dataFrame
data_set = data_set.join(one_hot)
data_set = data_set.drop('month',axis=1)




print(list(data_set))
"""
#data_set.education = dummies
enc = OneHotEncoder()
enc.fit(dummies)
print(enc.n_values_)
print(enc.feature_indices_)
enc.transform([[0, 1, 1, 0]]).toarray()
"""





# split up the data set 
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

data_set.to_csv("../data/afterTransform.csv", index = False)