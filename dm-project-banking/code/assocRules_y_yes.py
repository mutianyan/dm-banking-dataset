# Association rules mining 

# find frequent itemsets : algorithms: Apriori, fp-growth
import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori


def getSupport(lst_check):
	for (lst_val, sup) in lst:
		if lst_check == lst_val:
			return sup

dataset = pd.read_csv('../data/bank-full.csv', delimiter = ';', nrows = 40000)
# filter out all none y=yes rows
dataset = dataset.loc[dataset['y'] == 'yes']
dataset.drop('y', inplace = True, axis = 1)
dataset.to_csv('../data/all_y_yes.csv')

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
new_df=pd.DataFrame(data = array, index = range(2896), columns= header)

# Apriori algo:

te = TransactionEncoder()
te_ary = te.fit(array).transform(array)

# (unnecessary step) convert back to dataframe from numpy array
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, use_colnames=True, min_support = 0.5)

# generate the length column 
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# sort by length of the frequent pattern
frequent_itemsets.sort_values('length', inplace = True, ascending = False)
frequent_itemsets.sort_values('support', inplace = True, ascending = False)

# selecting by pandas entries
frequent_itemsets_drawing = frequent_itemsets[ (frequent_itemsets['length'] >= 2)]

# save the file 
frequent_itemsets_drawing.to_csv("../data/FrquentPatterns_y_yes.csv")
frozen_set=frequent_itemsets_drawing['itemsets']

frequent_itemsets_drawing_list = [list(x) for x in frozen_set if len(x) >= 2 ]
#print(frequent_itemsets_drawing_list)

#frequent_itemsets_drawing_list_of_tuple = list(map(tuple, frequent_itemsets_drawing_list))
#print(frequent_itemsets_drawing_list_of_tuple)

########## draw #########

# find all the attributes appears, assemble them in a list
attributes_drawing_list = []
for lst in frequent_itemsets_drawing_list:
	for val in lst:
		if val not in attributes_drawing_list:
			attributes_drawing_list.append(val)

#print(attributes_drawing_list)

# generate a dictionary, {[poutcome_1, ...] : support values}

support_vals = frequent_itemsets_drawing['support']

lst =list(zip(frequent_itemsets_drawing_list, support_vals))


# store in to csv
I = pd.Index(attributes_drawing_list, name="rows")
C = pd.Index(frequent_itemsets_drawing_list, name="columns")
d_f = pd.DataFrame(index=C, columns=I)
#print(d_f)
#d_f.a.fillna(value=0, inplace=True)
# write in the values
for fp, index in zip(frequent_itemsets_drawing_list, range(len(frequent_itemsets_drawing_list))):
	for item in fp:
		d_f.loc[index,item] = getSupport(fp)
		#print(1)

#d_f.loc[['loan_no', 'default_no']] = 0

d_f.to_csv("../graphs/frequent_patterns_y_yes.csv")

# generate strong association rules

# metric = lift ... 2 measures lift and confidence
rules = association_rules(frequent_itemsets)
rules.to_csv("../data/AssociationRules_y_yes.csv")

# selecting rules 
# using pandas filter
#only = rules[rules['consequents'] == frozenset({'y_no'}) ]
#print(only)

