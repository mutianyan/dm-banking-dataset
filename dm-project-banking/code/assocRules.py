# Association rules mining 

# find frequent itemsets : algorithms: Apriori, fp-growth
import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

dataset = pd.read_csv('../data/bank-full.csv', delimiter = ';', nrows = 40000)
print(dataset[:0])

header = list(dataset)
array = dataset.values

# convert the dataframe attribute tyoe from int to str
dataset[["age", "balance", "day", "duration", "campaign", "pdays", "previous"]] = dataset[["age", "balance", "day", "duration", "campaign", "pdays", "previous"]].astype(str) 

# attach the column name as prefix for each values
for col_name, column, col_index in zip (header, array.T, range(len(array.T))): # col index
	#print(col_index)
	for value, row in zip(column, range(len(array))):
		array[row][col_index] = col_name +"_"+ str(value) + "" 

# convert back to data frame
new_df=pd.DataFrame(data = array, index = range(40000), columns= header)

# Apriori algo:

te = TransactionEncoder()
te_ary = te.fit(array).transform(array)

# (unnecessary step) convert back to dataframe from numpy array
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# sort by length of the frequent pattern
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# selecting 
frequent_itemsets[ (frequent_itemsets['length'] == 2) &
                   (frequent_itemsets['support'] >= 0.8) ]
# selecting using pandas entries
#frequent_itemsets[ frequent_itemsets['itemsets'] == {'Onion', 'Eggs'} ]

# save the file 
frequent_itemsets.to_csv("../data/FrquentPatterns.csv")

# generate strong association rules

# metric = lift ... 2 measures lift and confidence
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules.to_csv("../data/AssociationRules.csv")

# selecting rules 

#rules[ ((rules['confidence'] > 0.75) &
#       (rules['lift'] > 1.2)) ]

#rules[rules['antecedents'] == {'Eggs', 'Kidney Beans'}]
# using pandas filter
only = rules[rules['consequents'] == frozenset({'y_yes'}) ]
print(only)

