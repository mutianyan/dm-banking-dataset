# mannually drop the extra columns on the expanded data -- selection
import pandas as pd
data_set = pd.read_csv('../data/all_after_expand_and_discretion_test.csv')

print(list(data_set))
"""
['age', 'default', 'balance', 'housing', 'loan', 'day', 'duration', 'campaign', 'pdays', 
'previous', 'y', 'deafult', 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown_job', 
'divorced', 'married', 'single', 'primary', 'secondary', 'tertiary', 'unknown_education', 
'cellular', 'telephone', 'unknown_contact', 'failure', 'other', 'success', 'unknown_poutcome', 
'apr', 'aug', 'dec', 'feb', 'jan', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep']
"""


# based on the rule 4 which has columns
#['month', 'duration', 'pdays', 'age', 'contact', 'housing', 'job', 'education', 'y']


new_df= data_set[['apr', 'aug', 'dec', 'feb', 'jan', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep',
'duration', 'pdays', 'age','cellular', 'telephone', 'unknown_contact', 'housing', 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown_job', 'primary', 'secondary', 'tertiary', 'unknown_education', 
'y']]


new_df.to_csv('../data/expanded_all_rule4.csv', index = False)
