import pandas as pd

class precess2limitedattr:
    def __init__(self,keylist):
        self.data_set = pd.read_csv('../data/bank-full.csv', delimiter=';')
        self.keylist = keylist
        # self.keylist = ['age','job','month','previous','poutcome', 'balance','duration', 'pdays']

    def getdata(self):
        dic = {}

        dic_job ={}
        dic_job['unknown']=0
        dic_job['management']=1
        dic_job['technician']=2
        dic_job['entrepreneur']=3
        dic_job['blue-collar']=4
        dic_job['retired']=5
        dic_job['admin']=6
        dic_job['services']=7
        dic_job['self-employed']=8
        dic_job['unemployed']=9
        dic_job['housemaid']=10
        dic_job['student']=11
        dic['job']=dic_job

        dic_marital={}
        dic_marital['married']=1
        dic_marital['single']=2
        dic_marital['divorced']=3
        dic['marital'] = dic_marital

        dic_education = {}
        dic_education['unknown']=0
        dic_education['primary']=1
        dic_education['secondary']=2
        dic_education['tertiary']=3
        dic['education']=dic_education

        dic_yesno ={}
        dic_yesno['yes']=1
        dic_yesno['no']=2
        dic['default']=dic_yesno

        dic['housing'] = dic_yesno
        dic['loan'] = dic_yesno

        dic_contact={}
        dic_contact['unknown']=0
        dic_contact['cellular']=1
        dic_contact['telephone']=2
        dic['contact']=dic_contact

        dic_month={}
        dic_month['jan']=1
        dic_month['feb']=2
        dic_month['mar']=3
        dic_month['apr']=4
        dic_month['mar']=5
        dic_month['jun']=6
        dic_month['jul']=7
        dic_month['aug']=8
        dic_month['sep']=9
        dic_month['oct']=10
        dic_month['nov']=11
        dic_month['dec']=12
        dic['month']=dic_month

        dic_poutcome={}
        dic_poutcome['unknown']=0
        dic_poutcome['success']=1
        dic_poutcome['failure']=2
        dic_poutcome['other']=3
        dic['poutcome']=dic_poutcome

        dic['y']=dic_yesno

        keys = self.data_set.keys()

        def transform(key):
            for i in range(len(self.data_set)):
                self.data_set[key][i]=dic[key][self.data_set[key][i]]

        label = transform('y')
        self.data_set.drop('y')


        for key in keys:
            if key not in self.keylist:
                self.data_set.drop(key)
                continue
            if key in ['job','marital','education','default','housing','loan','contact','month','poutcome']:
                transform(key)
                continue
            if key == 'age':
                self.data_set[key] = (self.data_set[key]-15)//5
                continue
            if key == 'balance':
                self.data_set[key] = self.data_set[key]//5000
                continue
            if key == 'duration':
                self.data_set[key] = self.data_set[key]//120
                continue
            if key == 'pdays':
                self.data_set[key] = self.ata_set[key]//30
                continue

        return self.data_set,label
