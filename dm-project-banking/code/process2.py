import pandas as pd
from collections import Counter
import numpy as np
def transforForamt(dataset, diction):
    for i in range(len(dataset)):
        dataset[i] = diction[dataset[i]]
    return dataset


data_set = pd.read_csv('../data/bank-full.csv',delimiter = ';')
def countFrac(counter,totalnum):
    dic={}
    for key in counter.keys():
        dic[key]=round(counter.get(key)/totalnum,4)
    return dic
def count_yes_rate(counter_all,counter_yes):
    dic = {}
    for key in counter_all.keys():
        if counter_yes.get(key):
            dic[key]=round(counter_yes.get(key)/counter_all.get(key),4)
        else:
            dic[key]=0.0
    return dic

data_set_yes = data_set[data_set['y']=='yes']
#age;"job";"marital";"education";"default";"balance";"housing";"loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"
age_count = Counter(data_set['age'])
age_count_yes = Counter(data_set_yes['age'])
job_count = Counter(data_set['job'])
job_count_yes = Counter(data_set_yes['job'])
marital_count = Counter(data_set['marital'])
marital_count_yes = Counter(data_set_yes['marital'])
education_count = Counter(data_set['education'])
education_count_yes = Counter(data_set_yes['education'])
default_count = Counter(data_set['default'])
default_count_yes = Counter(data_set_yes['default'])
balance_count = Counter(data_set['balance'])
balance_count_yes = Counter(data_set_yes['balance'])
housing_count = Counter(data_set['housing'])
housing_count_yes = Counter(data_set_yes['housing'])
loan_count = Counter(data_set['loan'])
loan_count_yes = Counter(data_set_yes['loan'])
contact_count = Counter(data_set['contact'])
contact_count_yes = Counter(data_set_yes['contact'])
day_count = Counter(data_set['day'])
day_count_yes = Counter(data_set_yes['day'])
month_count = Counter(data_set['month'])
month_count_yes = Counter(data_set_yes['month'])
duration_count = Counter(data_set['duration'])
duration_count_yes = Counter(data_set_yes['duration'])
campaign_count = Counter(data_set['campaign'])
campaign_count_yes = Counter(data_set_yes['campaign'])
pdays_count = Counter(data_set['pdays'])
pdays_count_yes = Counter(data_set_yes['pdays'])
previous_count = Counter(data_set['previous'])
previous_count_yes = Counter(data_set_yes['previous'])
poutcome_count = Counter(data_set['poutcome'])
poutcome_count_yes = Counter(data_set_yes['poutcome'])
y_count=Counter(data_set['y'])

totalnum = len(data_set)

print("age_count: ")
print(countFrac(age_count,len(data_set)))
print(count_yes_rate(age_count,age_count_yes))
print("job_count: ")
print(countFrac(job_count,len(data_set)))
print(count_yes_rate(job_count,job_count_yes))
print("marital_count: ")
print(countFrac(marital_count,len(data_set)))
print(count_yes_rate(marital_count,marital_count_yes))
print("education_count: ")
print(countFrac(education_count,len(data_set)))
print(count_yes_rate(education_count,education_count_yes))
print("default_count: ")
print(countFrac(default_count,len(data_set)))
print(count_yes_rate(default_count,default_count_yes))
print("balance_count: ")
print(countFrac(balance_count,len(data_set)))
print(count_yes_rate(balance_count,balance_count_yes))
print("housing_count: ")
print(countFrac(housing_count,len(data_set)))
print(count_yes_rate(housing_count,housing_count_yes))
print("loan_count: ")
print(countFrac(loan_count,len(data_set)))
print(count_yes_rate(loan_count,loan_count_yes))
print("contact_count: ")
print(countFrac(contact_count,len(data_set)))
print(count_yes_rate(contact_count,contact_count_yes))
print("day_count: ")
print(countFrac(day_count,len(data_set)))
print(count_yes_rate(day_count,day_count_yes))
print("month_count: ")
print(countFrac(month_count,len(data_set)))
print(count_yes_rate(month_count,month_count_yes))
print("duration_count: ")
print(countFrac(duration_count,len(data_set)))
print(count_yes_rate(duration_count,duration_count_yes))
print("campaign_count: ")
print(countFrac(campaign_count,len(data_set)))
print(count_yes_rate(campaign_count,campaign_count_yes))
print("pdays_count: ")
print(countFrac(pdays_count,len(data_set)))
print(count_yes_rate(pdays_count,pdays_count_yes))
print("previous_count: ")
print(countFrac(previous_count,len(data_set)))
print(count_yes_rate(previous_count,previous_count_yes))
print("poutcome_count: ")
print(countFrac(poutcome_count,len(data_set)))
print(count_yes_rate(poutcome_count,poutcome_count_yes))
print("y_count")
print(countFrac(y_count,len(data_set)))


def convert2Exl(counter_all,counter_yes,totalnum,title):
    dic={}
    for key in counter_all:
        val=[]
        val.append(round(counter_all.get(key)/totalnum,4))
        if counter_yes.get(key):
            val.append(round(counter_yes.get(key)/counter_all.get(key),4))
        else:
            val.append(0.0)
        dic[key]=val
    df=pd.DataFrame(dic)
    df.to_excel(title)


convert2Exl(job_count,job_count_yes,len(data_set),'job_count.xls')
convert2Exl(marital_count,marital_count_yes,len(data_set),'marital_count.xls')
convert2Exl(education_count,education_count_yes,len(data_set),'education_count.xls')
convert2Exl(default_count,default_count_yes,len(data_set),'default_count.xls')
convert2Exl(housing_count,housing_count_yes,len(data_set),'housing_count.xls')
convert2Exl(loan_count,loan_count_yes,len(data_set),'loan_count.xls')
convert2Exl(contact_count,contact_count_yes,len(data_set),'contact_count.xls')
convert2Exl(day_count,day_count_yes,len(data_set),'day_count.xls')
convert2Exl(month_count,month_count_yes,len(data_set),'month_count.xls')
convert2Exl(campaign_count,campaign_count_yes,len(data_set),'campaign_count.xls')
convert2Exl(previous_count,previous_count_yes,len(data_set),'previous_count.xls')
convert2Exl(poutcome_count,poutcome_count_yes,len(data_set),'poutcome_count.xls')
convert2Exl(y_count,day_count_yes,len(data_set),'y_count.xls')


age=(data_set['age']-15)//5
age_count=Counter(age)
age_yes= (data_set_yes['age']-15)//5
age_count_yes=Counter(age_yes)
convert2Exl(age_count,age_count_yes,len(data_set),'age_count.xls')

balance = (data_set['balance'])//5000
balance_count=Counter(balance)
balance_yes = data_set_yes['balance']//5000
balance_count_yes = Counter(balance_yes)
convert2Exl(balance_count,balance_count_yes,len(data_set),'balance_count.xls')

duration = data_set['duration']//120
duration_count = Counter(duration)
duration_yes = data_set_yes['duration']//120
duration_count_yes = Counter(duration_yes)
convert2Exl(duration_count,duration_count_yes,len(data_set),'duration_count.xls')

pdays = data_set['pdays']//30
pdays_count = Counter(pdays)
pdays_yes = data_set_yes['pdays']//30
pdays_count_yes = Counter(pdays_yes)
convert2Exl(pdays_count,pdays_count_yes,len(data_set),'pdays_count.xls')

def findbestattr(counteralllist,counteryeslist,attrlist):
    minlist={}
    maxlist={}
    difflist={}
    for i in range(len(attrlist)):
    # for counteryes, counterall,name in counteralllist,counteryeslist,attrlist:
        vallist=[]
        for key in counteralllist[i].keys():
            if counteryeslist[i].get(key):
                val=round(counteryeslist[i].get(key) / counteralllist[i].get(key), 4)
            else:
                val=0.0
            vallist.append(val)
        minlist[attrlist[i]]=(np.min(vallist))
        maxlist[attrlist[i]]=(np.max(vallist))
        difflist[attrlist[i]]=round((np.max(vallist)-np.min(vallist)),4)
    return minlist,maxlist,difflist

counteralllist = []
counteryeslist = []
attrlist = []
attrlist.append('age')
counteralllist.append(age_count)
counteryeslist.append(age_count_yes)
attrlist.append('job')
counteralllist.append(job_count)
counteryeslist.append(job_count_yes)
attrlist.append('marital')
counteralllist.append(marital_count)
counteryeslist.append(marital_count_yes)
attrlist.append('education')
counteralllist.append(education_count)
counteryeslist.append(education_count_yes)
attrlist.append('default')
counteralllist.append(default_count)
counteryeslist.append(default_count_yes)
attrlist.append('housing')
counteralllist.append(housing_count)
counteryeslist.append(housing_count_yes)
attrlist.append('loan')
counteralllist.append(loan_count)
counteryeslist.append(loan_count_yes)
attrlist.append('contact')
counteralllist.append(contact_count)
counteryeslist.append(contact_count_yes)
attrlist.append('day')
counteralllist.append(day_count)
counteryeslist.append(day_count_yes)
attrlist.append('month')
counteralllist.append(month_count)
counteryeslist.append(month_count_yes)
attrlist.append('campaign')
counteralllist.append(campaign_count)
counteryeslist.append(campaign_count_yes)
attrlist.append('previous')
counteralllist.append(previous_count)
counteryeslist.append(previous_count_yes)
attrlist.append('poutcome')
counteralllist.append(poutcome_count)
counteryeslist.append(poutcome_count_yes)
attrlist.append('balance')
counteralllist.append(balance_count)
counteryeslist.append(balance_count_yes)
attrlist.append('duration')
counteralllist.append(duration_count)
counteryeslist.append(duration_count_yes)
attrlist.append('pdays')
counteralllist.append(pdays_count)
counteryeslist.append(pdays_count_yes)

minlist,maxlist,difflist = findbestattr(counteralllist,counteryeslist,attrlist)
print("diff")
print(difflist)
print("max")
print(maxlist)
print("min")
print(minlist)

dic={"Female":1,"Male":0}
data_set['gender'] = transforForamt(data_set['gender'],dic)


dic_2={"Yes":1,"No":0}
data_set['Partner'] = transforForamt(data_set['Partner'],dic_2)
data_set['Dependents'] = transforForamt(data_set['Dependents'],dic_2)
data_set['PhoneService'] = transforForamt(data_set['PhoneService'],dic_2)
data_set['PaperlessBilling'] = transforForamt(data_set['PaperlessBilling'],dic_2)
data_set['Churn'] = transforForamt(data_set['Churn'],dic_2)

dic_3={"No internet service":-1,"Yes":1, "No":0}
data_set['OnlineSecurity'] = transforForamt(data_set['OnlineSecurity'],dic_3)
data_set['OnlineBackup'] = transforForamt(data_set['OnlineBackup'],dic_3)
data_set['DeviceProtection'] = transforForamt(data_set['DeviceProtection'],dic_3)
data_set['TechSupport'] = transforForamt(data_set['TechSupport'],dic_3)
data_set['StreamingTV'] = transforForamt(data_set['StreamingTV'],dic_3)
data_set['StreamingMovies'] = transforForamt(data_set['StreamingMovies'],dic_3)

dic_4={"No":0,"DSL":1, "Fiber optic":3}
data_set['InternetService'] = transforForamt(data_set['InternetService'],dic_4)


dic_5={"Month-to-month":1,"One year":2, "Two year":3}
data_set['Contract'] = transforForamt(data_set['Contract'],dic_5)

dic_6={"Electronic check":1,"Mailed check":2, "Bank transfer (automatic)":3, "Credit card (automatic)":4}
data_set['PaymentMethod'] = transforForamt(data_set['PaymentMethod'],dic_6)

dic_7={"No phone service":-1,"Yes":1, "No":0}
data_set['MultipleLines'] = transforForamt(data_set['MultipleLines'],dic_7)



data_suffle = data_set.sample(frac=1)
train_ratio = 0.638
train_idx = int(train_ratio * data_suffle.shape[0])

train_data = data_suffle[0:train_idx]
test_data = data_suffle[train_idx+1:-1]
dataframe_train = pd.DataFrame(train_data)
dataframe_train.to_csv("data/train.csv")
dataframe_test = pd.DataFrame(test_data)
dataframe_test.to_csv("data/test.csv")




