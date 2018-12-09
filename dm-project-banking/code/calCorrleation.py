# calcaute the correlation between all the "rolled-up" attributes and the Churn value
# and then use Navie Bayesian to classify
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import chisquare
import matplotlib.pyplot as plt

df = pd.read_csv('../data/bank-full.csv',delimiter = ';')
all_column = list(df)

def calChi2(col_name):
	df_1 = df.filter([col_name,'y'], axis=1)
	df_4 =pd.crosstab(df_1[col_name], df_1.y)
	#print('the numberic distribution table\n', df_4)
	(x,_,_,_) = st.chi2_contingency(pd.crosstab(df_1[col_name], df_1.y))
	print('chi2:', x, 'for', col_name)

#all_column=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
all_column=[ 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',  'month', 'campaign', 'pdays', 'previous', 'poutcome', 'y']


# not function for numeric data yet: age, balanace, day, duration
# also need to change to categorical values to interger representation
print(all_column)
#map(lambda x:x.calChi2(x),all_column)

for obj in all_column:
       calChi2(obj)