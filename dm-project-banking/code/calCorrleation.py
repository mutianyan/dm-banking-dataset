# calcaute the correlation between all the "rolled-up" attributes and the Churn value
# and then use Navie Bayesian to classify
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import operator
from decimal import Decimal

#df = pd.read_csv('../data/afterDiscretized.csv')
df = pd.read_csv('../data/afterTransform.csv')

# cotinuous values to discrete values
# discretization

for key in ['age', 'balance', 'duration', 'pdays']:
    if key == 'age':
        df[key] = (df[key]-15)//5
        continue
    if key == 'balance':
        df[key] = df[key]//5000
        continue
    if key == 'duration':
        df[key] = df[key]//120
        continue
    if key == 'pdays':
        df[key] = df[key]//30
        continue

df.to_csv('../data/all_after_expand_and_discretion.csv')


df = pd.read_csv('../data/all_after_discretion_of_continuous_val.csv')
dic = {}
dic_fd = {}



# correaltion between all the attributes and y
def calChi2(col_name):
	# subset columns of dataframe
	df_1 = df.filter([col_name,'y'], axis=1)
	# compute a single cross-tabulation of two (or more) factors
	df_4 =pd.crosstab(df_1[col_name], df_1.y)
	#print('the numberic distribution table\n', df_4)
	# test of independence of variables in a contingency table (index = all the columns, columns = y)
	(chi1,p_val,dof,_) = st.chi2_contingency(pd.crosstab(df_1[col_name], df_1.y))
	#print('chi2:', x, 'for', col_name
	dic[col_name] = p_val #Decimal(p_val).quantize(Decimal('0.01'))
	dic_fd[col_name] = dof

all_column=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
#all_column=[ 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',  'month', 'campaign', 'pdays', 'previous', 'poutcome']


# not function for numeric data yet: age, balanace, day, duration
# also need to change to categorical values to interger representation
#print(all_column)
#map(lambda x:x.calChi2(x),all_column)

for obj in all_column:
       calChi2(obj)

sorted_dic = sorted(dic.items(), key=operator.itemgetter(1))
print(sorted_dic)
list_by_p_val = []
for (item,_) in sorted_dic:
	list_by_p_val.append(item)
print(list_by_p_val)
sorted_dic_fd = sorted(dic_fd.items(), key=operator.itemgetter(1))
print(sorted_dic_fd)



# correlation between attributes each other

#matplotlib inline


df.drop(df.columns[0], axis = 1, inplace = True)
print(df.corr().shape)
print(list(df))
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111) #111
    cmap = cm.get_cmap('jet', 30) # 30
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap) # Compute pairwise correlation of columns
    ax1.grid(True)
    plt.title('Features Correlation')
    labels = list(df)
    #labels=['Sex','Length','Diam','Height','Whole','Shucked','Viscera','Shell','Rings',]
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    xticklabels = list(df)
    yticklabels = list(df)
    plt.xticks(np.arange(len(xticklabels)) + 0.5, xticklabels, rotation = 45)
    plt.yticks(np.arange(len(yticklabels)) + 0.5, yticklabels)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    #plt.show()

correlation_matrix(df)




