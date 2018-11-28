import csv
import pandas as pd

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

data_set.to_csv("../data/afterTransform.csv", index = False)