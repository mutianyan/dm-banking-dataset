# expand columns of the selected dataset
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
# avoid splitting the dataset again

# expanding begins
#print(col_names)
#['month', 'duration', 'pdays', 'age', 'contact', 'housing', 'job', 'education', 'y']

for filename in ['all_rule4_train', 'all_rule4_test']:
	data_set = pd.read_csv('../data/%s.csv' % filename)
	col_names = list(data_set)


	# need to convet type and then add prefix to the values
	# unknown is not existing in the dataset anymore
	if 'job' in col_names:
		data_set.loc[data_set['job'] == 'unknown', 'job'] = 'unknown_job'
	if 'education' in col_names:
		data_set.loc[data_set['education'] == 'unknown', 'education'] = 'unknown_education'
	if 'contact' in col_names:
		data_set.loc[data_set['contact'] == 'unknown', 'contact'] = 'unknown_contact'
	if 'poutcome' in col_names:
		data_set.loc[data_set['poutcome'] == 'unknown', 'poutcome'] = 'unknown_poutcome'

	if 'job' in col_names:
		s = data_set.job
		one_hot=pd.get_dummies(s) # return type is dataFrame
		data_set = data_set.join(one_hot)
		data_set = data_set.drop('job',axis=1)
	if 'marital' in col_names:
		s = data_set.marital
		one_hot=pd.get_dummies(s) # return type is dataFrame
		data_set = data_set.join(one_hot)
		data_set = data_set.drop('marital',axis=1)
	if 'eduction' in col_names:
		s = data_set.education
		one_hot=pd.get_dummies(s) # return type is dataFrame
		data_set = data_set.join(one_hot)
		data_set = data_set.drop('education',axis=1)
	if 'contact' in col_names:
		s = data_set.contact
		one_hot=pd.get_dummies(s) # return type is dataFrame
		data_set = data_set.join(one_hot)
		data_set = data_set.drop('contact',axis=1)
	if 'poutcome' in col_names:
		s = data_set.poutcome
		one_hot=pd.get_dummies(s) # return type is dataFrame
		data_set = data_set.join(one_hot)
		data_set = data_set.drop('poutcome',axis=1)

	# ? depends on how to treat the month, in order or no
	if 'month' in col_names:
		s = data_set.month
		one_hot=pd.get_dummies(s) # return type is dataFrame
		data_set = data_set.join(one_hot)
		data_set = data_set.drop('month',axis=1)
	data_set.to_csv('../data/expanded_%s.csv' % filename, index = False)


