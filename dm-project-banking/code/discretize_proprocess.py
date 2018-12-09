# this file produces the second version preprocessed file with all the categorical values being discretized (binary)
# the file created is named "afterDiscretized.csv"
import csv
import pandas as pd
import numpy as np

# bank.csv from May 2008 to November 2010

data_set = pd.read_csv('../data/bank-full.csv',delimiter = ';')

def transforForamt(dataset, diction):
    for i in range(len(dataset)):
        dataset[i] = diction[dataset[i]]
    return dataset

# need to think of a way to treat numeric attibutes like campaign...

dic={"yes":1,"no":0}
data_set['deafult'] = transforForamt(data_set['default'],dic)
data_set['housing'] = transforForamt(data_set['housing'],dic)
data_set['loan'] = transforForamt(data_set['loan'],dic)
data_set['y'] = transforForamt(data_set['y'],dic)


dic_1={"admin.":1,"unknown":0, "unemployed":2, "management":3, "housemaid":4, "entrepreneur":5, "student":6, "blue-collar":7, "self-employed":8,"retired":9,"technician": 10,"services": 11}
dic_2={"married":1,"divorced":2,"single":3}
dic_3={"secondary":2, "primary":1, "tertiary":3, "unknown":0}
dic_4={"telephone":1,"unknown":0, "cellular":2}
dic_5={"jan":1,"feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
dic_6={"other":3,"failure":1,"success":2, "unknown":0}


data_set['job'] = transforForamt(data_set['job'],dic_1)
data_set['marital'] = transforForamt(data_set['marital'],dic_2)
data_set['education'] = transforForamt(data_set['education'],dic_3)
data_set['contact'] = transforForamt(data_set['contact'],dic_4)
data_set['month'] = transforForamt(data_set['month'],dic_5)
data_set['poutcome'] = transforForamt(data_set['poutcome'],dic_6)

# categorical and continuous variables
#https://scikit-learn.org/stable/modules/preprocessing.html

# dont save the index
data_set.to_csv("../data/afterDiscretized.csv", index = False)