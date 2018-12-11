import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('../data/all_y_yes.csv')

age = df['age']
plt.hist(age)
plt.xlabel('ages')
plt.ylabel('number of samples')
plt.title(r'The age univariate distribution in the dataset')
plt.grid(True)
plt.show() 


balance = df['balance']
plt.scatter(age, balance)
plt.xlabel('age')
plt.ylabel('current balance in the bank')
plt.title(r'The scatter plot of age and balance')
plt.grid(True)
plt.show() 