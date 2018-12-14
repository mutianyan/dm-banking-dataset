import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
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

def computeEntropy(counter_all,counter_yes):
    splitInfo=0
    total = sum(counter_all.values())
    gain = 0
    for key in counter_all.keys():
        pi=counter_all[key]/total
        gain += -pi*np.log(pi)
        if counter_yes[key]:
            p=counter_yes[key]/counter_all[key]
            info_a=-p*np.log(p)-(1-p)*np.log(1-p)
        else:
            info_a=0
        splitInfo += counter_all[key]/total*info_a
    return gain/splitInfo





def convert2Exl(counter_all,counter_yes,totalnum,title,keyOrder,graphtitle,graphname):
    print(title+"  "+str(computeEntropy(counter_all,counter_yes)))

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
    bb=[]
    bt=[]

    ind = np.arange(len(keyOrder))
    for i in keyOrder:

        bt.append(dic[i][0]-dic[i][1]*dic[i][0])
        bb.append(dic[i][1]*dic[i][0])

    # plt.figure()
    # p1 = plt.bar(ind, bb)
    # p2 = plt.bar(ind, bt,
    #              bottom=bb)
    #
    # plt.ylabel(property)
    # plt.title(graphtitle)
    # plt.xticks(ind, keyOrder)
    # plt.xticks(rotation=75)
    # plt.legend((p1[0], p2[0]), ('P(y|x)', 'P(x)'))
    # plt.savefig(graphname)
    #
    # plt.show()

    bl=[]
    br=[]
    for key in keyOrder:
        bl.append(dic[key][0])
        br.append(dic[key][1])
    x = list(range(len(keyOrder)))
    width=0.3
    n=len(keyOrder)
    plt.bar(x,bl,width=width,label='P(x)',fc='b')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x,br,width=width,label='P(y|x)',tick_label = keyOrder,fc='g')
    plt.xticks(rotation=75)
    plt.legend()
    plt.title(graphtitle)
    plt.ylabel(property)
    plt.savefig("lr"+graphname)
    plt.show()


    # name_list = ['Monday', 'Tuesday', 'Friday', 'Sunday']
    # num_list = [1.5, 0.6, 7.8, 6]
    # num_list1 = [1, 2, 3, 1]
    # x = list(range(len(num_list)))
    # total_width, n = 0.8, 2
    # width = total_width / n
    #
    # plt.bar(x, num_list, width=width, label='boy', fc='y')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, num_list1, width=width, label='girl', tick_label=name_list, fc='r')
    # plt.legend()



convert2Exl(job_count,job_count_yes,len(data_set),'job_count.xls',
            ['entrepreneur','management','admin.','self-employed','technician','services','blue-collar','housemaid','retired','student','unemployed','unknown'],
            'job attribute distribution','job attribute distribution.png')
convert2Exl(marital_count,marital_count_yes,len(data_set),'marital_count.xls',['married','single','divorced'],
            'marital attribute distribution','marital attribute distribution.png')
convert2Exl(education_count,education_count_yes,len(data_set),'education_count.xls',['tertiary','secondary','primary','unknown'],
            'education attribute distribution','education attribute distribution.png')
convert2Exl(default_count,default_count_yes,len(data_set),'default_count.xls',['yes','no'],'default attribute distribution','default attribute distribution.png')
convert2Exl(housing_count,housing_count_yes,len(data_set),'housing_count.xls',['yes','no'],'housing attribute distribution','housing attribute distribution.png')
convert2Exl(loan_count,loan_count_yes,len(data_set),'loan_count.xls',['yes','no'],'loan attribute distribution','loan attribute distribution.png')
convert2Exl(contact_count,contact_count_yes,len(data_set),'contact_count.xls',['cellular','telephone','unknown'],'contact attribute distribution','contact attribute distribution.png')
convert2Exl(day_count,day_count_yes,len(data_set),'day_count.xls',np.arange(1,32,1).tolist(),'day attribute distribution','day attribute distribution.png')
convert2Exl(month_count,month_count_yes,len(data_set),'month_count.xls',['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],'month attribute distribution','month attribute distribution.png')
convert2Exl(campaign_count,campaign_count_yes,len(data_set),'campaign_count.xls',sorted(list(campaign_count.keys())),'campaign attribute distribution','campaign attribute distribution.png')
convert2Exl(previous_count,previous_count_yes,len(data_set),'previous_count.xls',sorted(list(previous_count.keys())),'previous attribute distribution','previous attribute distribution.png')
convert2Exl(poutcome_count,poutcome_count_yes,len(data_set),'poutcome_count.xls',['success','failure','other','unknown'],'poutcome attribute distribution','poutcome attribute distribution.png')
# convert2Exl(y_count,day_count_yes,len(data_set),'y_count.xls')


age=(data_set['age']-15)//5
age_count=Counter(age)
age_yes= (data_set_yes['age']-15)//5
age_count_yes=Counter(age_yes)
convert2Exl(age_count,age_count_yes,len(data_set),
            'age_count.xls',sorted(list(age_count.keys())),
            'age attribute distribution','age attribute distribution.png')

balance = (data_set['balance'])//5000
balance_count=Counter(balance)
balance_yes = data_set_yes['balance']//5000
balance_count_yes = Counter(balance_yes)
convert2Exl(balance_count,balance_count_yes,len(data_set),'balance_count.xls',sorted(list(balance_count.keys())),'balance attribute distribution(unit: 5000$)','balance attribute distribution.png')

duration = data_set['duration']//120
duration_count = Counter(duration)
duration_yes = data_set_yes['duration']//120
duration_count_yes = Counter(duration_yes)
convert2Exl(duration_count,duration_count_yes,len(data_set),'duration_count.xls',sorted(list(duration_count.keys())),'duration attribute distribution(unit: 2 minute)','duration attribute distribution.png')

pdays = data_set['pdays']//30
pdays_count = Counter(pdays)
pdays_yes = data_set_yes['pdays']//30
pdays_count_yes = Counter(pdays_yes)
convert2Exl(pdays_count,pdays_count_yes,len(data_set),'pdays_count.xls',sorted(list(pdays_count.keys())),'pdays attribute distribution(unit: month)','pdays attribute distribution.png')

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



