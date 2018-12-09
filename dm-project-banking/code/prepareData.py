# calculate a 5-number summary
from numpy import percentile
from numpy.random import rand
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

data_set = pd.read_csv('../data/bank-full.csv',delimiter = ';')

# for numeric continuous type of data
columns  = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

for col in columns:
	data = data_set[col]
	# calculate quartiles
	quartiles = percentile(data, [25, 50, 75])
	# calculate min/max
	data_min, data_max = data.min(), data.max()
	# print 5-number summary
	print('For column ', col)
	print('Min: %.3f' % data_min)
	print('Q1: %.3f' % quartiles[0])
	print('Median: %.3f' % quartiles[1])
	print('Q3: %.3f' % quartiles[2])
	print('Max: %.3f' % data_max)
	print('\n')
	

	plt.boxplot(data)
	plt.title('Box Plot for attribute (%s)' % col)
	#plt.show()
	plt.savefig('../graphs/Boxplot_%s.png' % col)

# draw box plot to visualize distribution

# discretion
X = data_set['age']
est = preprocessing.KBinsDiscretizer(n_bins=9, encode='ordinal').fit(X)
est.transform(X) 