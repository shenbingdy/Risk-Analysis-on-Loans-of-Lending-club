## Import the necessary packages
import requests, json, os, sys, time, re, math
from datetime import datetime
from sqlalchemy import *
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

#split fuction: split the long list
def split_list(list_long, m):
    New_list=[]
    n=len(list_long)
    if n%m==0:
        for i in range(int(n/m)):
            New_list.append(list_long[i*m:(i+1)*m])
    else:
        for i in range(int(n/m)):
            New_list.append(list_long[i*m:(i+1)*m])
        New_list.append(list_long[(i+1)*m:n])
    return New_list

# crawble from the web 
# Get the data from web (1) get the data alive by api(2) download the 2014 data for model training  

# 1) get the data alive by api

url='https://api.lendingclub.com/api/investor/v1/loans/listing'
para={'Authorization':'9SRITvnLWlYUsUWZW0h6HiToqVg='}
r=requests.get(url,headers=para).json()
r_new={}
for i in range(len(r.get('loans'))):
    r_new.update({str(i):r.get('loans')[i]})
path_list='C:/Users/shenbingdy/Desktop/datalab/finTech/data/loadnlist.csv'
df_loans.to_csv(path_list)

# 2) download the 2014 data for model training  
path_list_2014='C:/Users/shenbingdy/Desktop/datalab/finTech/data/LoanStats3c_securev1.csv'
df_loans2014=pd.read_csv(path_list_2014,header=1)
df_loans=pd.read_csv(path_list)

# 3) analysis of the feature to get the same feature for both data

# To compare we need to transform the columns name

# for loans list we need to lower the str of column
df_loans_columns_N= [x.lower() for x in list(df_loans.columns)]
#for 2014 we nned to get rid of '-'
df_loans_columns2014_N=[x.replace('_','') for x in list(df_loans2014.columns)]
dict_col=dict(zip(df_loans.columns,df_loans_columns_N))
df_loans.rename(columns=dict_col, inplace=True)
dict_2014_col=dict(zip(df_loans2014.columns,df_loans_columns2014_N))
df_loans2014.rename(columns=dict_2014_col, inplace=True)

# #We get the common and different features 
feature=[ x for x in df_loans_columns_N if  x in df_loans_columns2014_N]
different=[ x for x in df_loans_columns_N if  x not in df_loans_columns2014_N]

# Sometimes, the label of columns changes but the columns still exist in old data
# 1) expd and acceprd can be transform to emp_length which ius in old data
# 2) loanamount is the same as loan_amnt
# 3) addzip is the same as zipcode
# 4) fundedamount is the same as fundedamnt
# 5) numacctsever120ppd is the same as num_accts_ever_120_pd
# 6) isincv is the same as verificationsatus ?
# 7) isincvjoint is the same as verificationsatusjoint ?

samekey=['loanamount','addrzip','fundedamount','numacctsever120ppd','isincv','isincvjoint']
samevalue=['loanamnt', 'zipcode','fundedamnt','numacctsever120pd','verificationstatus','verificationstatusjoint']
same_dict=dict(zip(samekey,samevalue))
feature1=samevalue+feature
feature1+=['loanstatus', 'issued']# issued split tain and test
feature2=samekey+feature

# EDA

# The loan status is the pridiction for what we want we need to select the status is charge off or fully paid for training 
# This time we just select month =36 data
df_2014=df_loans2014.query("term==' 36 months'") 
df_2014=df_2014.query("(loanstatus=='Fully Paid')|(loanstatus=='Charged Off')") 
df_2014=df_2014[feature1]
df_loans_now=df_loans[feature2]
df_loans_now['loanstatus']='current'
df_loans_now['issued']='now'
df_loans_now.columns=df_2014.columns

#check the columns of non value
Non_value=['inqfi', 'ilutil', 'openrv12m', 'dtijoint',  'totalcutl',  'inqlast12m', 'openrv24m',  'openacc6m', 'totalbalil', 'allutil', 'openil24m', 'openil12m', 'mthssincercntil', 'memberid', 'maxbalbc', 'annualincjoint', 'verificationstatusjoint']
df_2014.drop(Non_value, inplace=True, axis=1)
df_loans_now.drop(Non_value, inplace=True, axis=1)

# check the features: continous and categorical 
Object=[]
Number=[]
for i in df_2014.columns:
    if df_2014[i].dtype=='object':
        Object+=[i]
    else:
        Number+=[i]   

continuous=[]
categorical=[]
for i in Number:
    if len(df_2014[i])<=25:
        categorical+=[i]
    else:
        continuous+=[i]   
#continuous=['totalilhighcreditlimit', 'collections12mthsexmed', 'mthssincerecentinq', 'totalacc', 'delinq2yrs', 'numbctl', 'numsats', 'numoprevtl', 'taxliens', 'pcttlnvrdlq', 'mthssincerecentbcdlq', 'mthssincelastmajorderog', 'mortacc', 'ficorangehigh', 'annualinc', 'mosinoldrevtlop', 'pubrecbankruptcies', 'numiltl', 'dti', 'numbcsats', 'numactvrevtl', 'inqlast6mths', 'numrevaccts', 'numtl30dpd', 'mthssincelastrecord', 'numtloppast12m', 'accnowdelinq', 'installment', 'delinqamnt', 'mthssincelastdelinq', 'revolbal', 'numtl120dpd2m', 'bcopentobuy', 'totcurbal', 'numactvbctl', 'bcutil', 'chargeoffwithin12mths', 'mosinrcntrevtlop', 'totcollamt', 'avgcurbal', 'mthssincerecentbc', 'numtl90gdpd24m', 'mosinoldilacct', 'numrevtlbalgt0', 'accopenpast24mths', 'openacc', 'mosinrcnttl', 'mthssincerecentrevoldelinq', 'tothicredlim', 'percentbcgt75', 'pubrec', 'totalbclimit', 'totalbalexmort', 'totalrevhilim', 'ficorangelow', 'loanamnt', 'fundedamnt', 'numacctsever120pd']
df_2014.reset_index(inplace=True)

# check the feature one by one

df_2014_N=pd.DataFrame(df_2014['id']) 
df_loans_now_N=pd.DataFrame(df_loans_now['id']) 
df_2014_N['loanstatus']=df_2014['loanstatus']
df_2014_N['loanstatus']=df_2014_N['loanstatus'].apply(lambda x: 1 if x=='Charged Off' else 0)

df_loans_now_N['loanstatus']=df_loans_now['loanstatus']

# check categorical feature

df_2014_N['intrate']=df_2014['intrate']
df_2014_N['intrate']=df_2014_N['intrate'].apply(lambda x: float(x.replace('%','')))

df_loans_now_N['intrate']=df_loans_now['intrate']


# check the grade and subgrade

df_2014_N['grade']=df_2014['grade']
grade_L=['A','B','C','D','E','F','G']
grade_N=[0,1,2,3,4,5,6]
df_2014_N['grade'].replace(grade_L,grade_N, inplace=True)

df_loans_now_N['grade']=df_loans_now['grade']
df_loans_now_N['grade'].replace(grade_L,grade_N, inplace=True)

df_2014_N['subgrade']=df_2014['subgrade']
subgrade=['A1', 'A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5']
subgrade_n=range(35)
df_2014_N['subgrade'].replace(subgrade, subgrade_n, inplace=True)

df_loans_now_N['subgrade']=df_loans_now['subgrade']
df_loans_now_N['subgrade'].replace(subgrade, subgrade_n, inplace=True)

# check the 
df_2014_N['emplength']=df_2014['emplength']
df_2014_N['emplength'].replace('n/a', np.nan, inplace=True)
df_2014_N['emplength'].replace('< 1 year', '0', inplace=True)
df_2014_N['emplength'].replace(to_replace='[^0-9]', value='', inplace=True, regex=True)
df_2014_N['emplength'].fillna(-9999, inplace=True)
df_2014_N['emplength']=df_2014_N['emplength'].astype(int)

df_loans_now_N['emplength']=df_loans_now['emplength']
df_loans_now_N['emplength']=df_loans_now_N['emplength'].apply(lambda x: 0 if pd.notnull(x) else x)


# In[33]:
df_2014_N['homeownership']=df_2014['homeownership']
df_loans_now_N['homeownership']=df_loans_now['homeownership']
df_2014_N['purpose']=df_2014['purpose']
df_loans_now_N['purpose']=df_loans_now['purpose']

# In[35]:
df_2014_N['addrstate']=df_2014['addrstate']
df_loans_now_N['addrstate']=df_loans_now['addrstate']

df_2014_N['initialliststatus']=df_2014['initialliststatus']
df_loans_now_N['initialliststatus']=df_loans_now['initialliststatus']

df_2014_N['emptitle']=df_2014['emptitle']
df_loans_now_N['emptitle']=df_loans_now['emptitle']

# In[38]:
df_2014_N['earliestcrline']=df_2014['earliestcrline']
df_2014_N['earliestcrline_month']=df_2014_N['earliestcrline'].apply(lambda x: (datetime.strptime(x.replace('-',' '), '%b %Y')).month)
df_2014_N['earliestcrline']=df_2014_N['earliestcrline'].apply(lambda x: datetime.today().year-(datetime.strptime(x.replace('-',' '), '%b %Y')).year)

df_loans_now_N['earliestcrline']=df_loans_now['earliestcrline']
df_loans_now_N['earliestcrline_month']=df_loans_now['earliestcrline'].apply(lambda x: pd.to_datetime(x).month)
df_loans_now_N['earliestcrline']=df_loans_now_N['earliestcrline'].apply(lambda x:datetime.today().year-pd.to_datetime(x).year)

# In[40]:
df_2014_N['revolutil']=df_2014['revolutil']
df_2014_N['revolutil']=df_2014_N['revolutil'].apply(lambda x: float(x.replace('%','')) if pd.notnull(x) else x)
df_loans_now_N['revolutil']=df_loans_now['revolutil']

df_2014_N['zipcode']=df_2014['zipcode']
df_loans_now_N['zipcode']=df_loans_now['zipcode']

df_2014_N['verificationstatus']=df_2014['verificationstatus'].apply(lambda x: x.lower())
df_loans_now_N['verificationstatus']=df_loans_now['verificationstatus'].apply(lambda x: x.replace('_',' ').lower())

df_2014_N['issued']=df_2014['issued']
df_2014_N['issued']=df_2014_N['issued'].apply(lambda x: (datetime.strptime(x.replace('-',' '), '%b %Y')).month)
df_loans_now_N['issued']=df_loans_now['issued']

# check the continuous features
continuous+=['revolutil','intrate'] #these two features are continuous

j=0
count=1
for i in range(len((split_list(continuous,6)[j])))[: : 2]:
    fig=plt.figure(figsize=(15,10))
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i])
    
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count+1)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i+1]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i+1])
     
    count+=2

split_list(continuous,6)[0]

df_2014_N[['totalilhighcreditlimit',
 'collections12mthsexmed',
 'mthssincerecentinq',
 'totalacc',
 'delinq2yrs',
 'numbctl']]=df_2014[['totalilhighcreditlimit',
 'collections12mthsexmed',
 'mthssincerecentinq',
 'totalacc',
 'delinq2yrs',
 'numbctl']]
df_loans_now_N[['totalilhighcreditlimit',
 'collections12mthsexmed',
 'mthssincerecentinq',
 'totalacc',
 'delinq2yrs',
 'numbctl']]=df_loans_now[['totalilhighcreditlimit',
 'collections12mthsexmed',
 'mthssincerecentinq',
 'totalacc',
 'delinq2yrs',
 'numbctl']]

j=1
count=1
for i in range(len((split_list(continuous,6)[j])))[: : 2]:
    fig=plt.figure(figsize=(15,10))
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i])
    
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count+1)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i+1]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i+1])
     
    count+=2

split_list(continuous,6)[1]

df_2014_N[['numsats',
 'numoprevtl',
 'taxliens',
 'pcttlnvrdlq',
 'mthssincerecentbcdlq',
 'mthssincelastmajorderog']]=df_2014[['numsats',
 'numoprevtl',
 'taxliens',
 'pcttlnvrdlq',
 'mthssincerecentbcdlq',
 'mthssincelastmajorderog']]
df_loans_now_N[['numsats',
 'numoprevtl',
 'taxliens',
 'pcttlnvrdlq',
 'mthssincerecentbcdlq',
 'mthssincelastmajorderog']]=df_loans_now[['numsats',
 'numoprevtl',
 'taxliens',
 'pcttlnvrdlq',
 'mthssincerecentbcdlq',
 'mthssincelastmajorderog']]
 
j=2
count=1
for i in range(len((split_list(continuous,6)[j])))[: : 2]:
    fig=plt.figure(figsize=(15,10))
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i])
    
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count+1)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i+1]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i+1])
     
    count+=2

split_list(continuous,6)[2]

df_2014_N[['mortacc',
 'ficorangehigh',
 'annualinc',
 'mosinoldrevtlop',
 'pubrecbankruptcies',
 'numiltl']]=df_2014[['mortacc',
 'ficorangehigh',
 'annualinc',
 'mosinoldrevtlop',
 'pubrecbankruptcies',
 'numiltl']]
df_loans_now_N[['mortacc',
 'ficorangehigh',
 'annualinc',
 'mosinoldrevtlop',
 'pubrecbankruptcies',
 'numiltl']]=df_loans_now[['mortacc',
 'ficorangehigh',
 'annualinc',
 'mosinoldrevtlop',
 'pubrecbankruptcies',
 'numiltl']]

j=3
count=1
for i in range(len((split_list(continuous,6)[j])))[: : 2]:
    fig=plt.figure(figsize=(15,10))
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i])
    
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count+1)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i+1]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i+1])
     
    count+=2

split_list(continuous,6)[3]

df_2014_N[['dti',
 'numbcsats',
 'numactvrevtl',
 'inqlast6mths',
 'numrevaccts',
 'numtl30dpd']]=df_2014[['dti',
 'numbcsats',
 'numactvrevtl',
 'inqlast6mths',
 'numrevaccts',
 'numtl30dpd']]
df_loans_now_N[['dti',
 'numbcsats',
 'numactvrevtl',
 'inqlast6mths',
 'numrevaccts',
 'numtl30dpd']]=df_loans_now[['dti',
 'numbcsats',
 'numactvrevtl',
 'inqlast6mths',
 'numrevaccts',
 'numtl30dpd']]

j=4
count=1
for i in range(len((split_list(continuous,6)[j])))[: : 2]:
    fig=plt.figure(figsize=(15,10))
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i])
    
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count+1)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i+1]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i+1])
     
    count+=2

split_list(continuous,6)[4]

df_2014_N[['mthssincelastrecord',
 'numtloppast12m',
 'accnowdelinq',
 'installment',
 'delinqamnt',
 'mthssincelastdelinq']]=df_2014[['mthssincelastrecord',
 'numtloppast12m',
 'accnowdelinq',
 'installment',
 'delinqamnt',
 'mthssincelastdelinq']]

df_loans_now_N[['mthssincelastrecord',
 'numtloppast12m',
 'accnowdelinq',
 'installment',
 'delinqamnt',
 'mthssincelastdelinq']]=df_loans_now[['dti',
 'numbcsats',
 'numactvrevtl',
 'inqlast6mths',
 'numrevaccts',
 'numtl30dpd']]

j=5
count=1
for i in range(len((split_list(continuous,6)[j])))[: : 2]:
    fig=plt.figure(figsize=(15,10))
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i])
    
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count+1)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i+1]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i+1])
     
    count+=2

split_list(continuous,6)[5]

df_2014_N[['revolbal',
 'numtl120dpd2m',
 'bcopentobuy',
 'totcurbal',
 'numactvbctl',
 'bcutil']]=df_2014[['revolbal',
 'numtl120dpd2m',
 'bcopentobuy',
 'totcurbal',
 'numactvbctl',
 'bcutil']]

df_loans_now_N[['revolbal',
 'numtl120dpd2m',
 'bcopentobuy',
 'totcurbal',
 'numactvbctl',
 'bcutil']]=df_loans_now[['revolbal',
 'numtl120dpd2m',
 'bcopentobuy',
 'totcurbal',
 'numactvbctl',
 'bcutil']]

j=6
count=1
for i in range(len((split_list(continuous,6)[j])))[: : 2]:
    fig=plt.figure(figsize=(15,10))
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i])
    
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count+1)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i+1]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i+1])
     
    count+=2

split_list(continuous,6)[6]

df_2014_N[['chargeoffwithin12mths',
 'mosinrcntrevtlop',
 'totcollamt',
 'avgcurbal',
 'mthssincerecentbc',
 'numtl90gdpd24m']]=df_2014[['chargeoffwithin12mths',
 'mosinrcntrevtlop',
 'totcollamt',
 'avgcurbal',
 'mthssincerecentbc',
 'numtl90gdpd24m']]

df_loans_now_N[['chargeoffwithin12mths',
 'mosinrcntrevtlop',
 'totcollamt',
 'avgcurbal',
 'mthssincerecentbc',
 'numtl90gdpd24m']]=df_loans_now[['chargeoffwithin12mths',
 'mosinrcntrevtlop',
 'totcollamt',
 'avgcurbal',
 'mthssincerecentbc',
 'numtl90gdpd24m']]

j=7
count=1
for i in range(len((split_list(continuous,6)[j])))[: : 2]:
    fig=plt.figure(figsize=(15,10))
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i])
    
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count+1)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i+1]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i+1])
     
    count+=2
split_list(continuous,6)[7]

df_2014_N[['mosinoldilacct',
 'numrevtlbalgt0',
 'accopenpast24mths',
 'openacc',
 'mosinrcnttl',
 'mthssincerecentrevoldelinq']]=df_2014[['mosinoldilacct',
 'numrevtlbalgt0',
 'accopenpast24mths',
 'openacc',
 'mosinrcnttl',
 'mthssincerecentrevoldelinq']]

df_loans_now_N[['mosinoldilacct',
 'numrevtlbalgt0',
 'accopenpast24mths',
 'openacc',
 'mosinrcnttl',
 'mthssincerecentrevoldelinq']]=df_loans_now[['mosinoldilacct',
 'numrevtlbalgt0',
 'accopenpast24mths',
 'openacc',
 'mosinrcnttl',
 'mthssincerecentrevoldelinq']]

j=8
count=1
for i in range(len((split_list(continuous,6)[j])))[: : 2]:
    fig=plt.figure(figsize=(15,10))
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i])
    
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count+1)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i+1]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i+1])
     
    count+=2

split_list(continuous,6)[8]

df_2014_N[['tothicredlim',
 'percentbcgt75',
 'pubrec',
 'totalbclimit',
 'totalbalexmort',
 'totalrevhilim']]=df_2014[['tothicredlim',
 'percentbcgt75',
 'pubrec',
 'totalbclimit',
 'totalbalexmort',
 'totalrevhilim']]

df_loans_now_N[['tothicredlim',
 'percentbcgt75',
 'pubrec',
 'totalbclimit',
 'totalbalexmort',
 'totalrevhilim']]=df_loans_now[['tothicredlim',
 'percentbcgt75',
 'pubrec',
 'totalbclimit',
 'totalbalexmort',
 'totalrevhilim']]

j=9
count=1
for i in range(4)[: : 2]:#(len((split_list(continuous,6)[j])))[: : 2]:
    fig=plt.figure(figsize=(15,10))
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i])
    
    plt.subplot(int(len(split_list(continuous,6)[j])/2),2, count+1)
    plt.violinplot(df_2014[(split_list(continuous,6)[j])[i+1]],showmeans=True)
    plt.title((split_list(continuous,6)[j])[i+1])
     
    count+=2

plt.violinplot(df_2014_N['intrate'],showmeans=True)

split_list(continuous,6)[9]

['ficorangelow',
 'loanamnt',
 'fundedamnt',
 'numacctsever120pd']

df_2014_N[['ficorangelow',
 'loanamnt',
 'fundedamnt',
 'numacctsever120pd']]=df_2014[['ficorangelow',
 'loanamnt',
 'fundedamnt',
 'numacctsever120pd']]

df_loans_now_N[['ficorangelow',
 'loanamnt',
 'fundedamnt',
 'numacctsever120pd']]=df_loans_now[['ficorangelow',
 'loanamnt',
 'fundedamnt',
 'numacctsever120pd']]


# merge all data for feature engineering
df_full=pd.concat([df_2014_N,df_loans_now_N],axis=0)


# treat the features addrstate,emptitle, and zipcode as frequncy counting

addrstate_fre=df_full.groupby('addrstate').size().reset_index()
addrstate_fre.columns=['addrstate','addrstate_fre']
df_full=pd.merge(df_full, addrstate_fre, how='left', on='addrstate')

emptitle_fre=df_full.groupby('emptitle').size().reset_index()
emptitle_fre.columns=['emptitle','emptitle_fre']
df_full=pd.merge(df_full, emptitle_fre, how ='left', on='emptitle')

zipcode=df_full.groupby('zipcode').size().reset_index()
zipcode.columns=['zipcode','zipcode_fre']
df_full=pd.merge(df_full, zipcode, how='left', on='zipcode')

# check the corrolation of continous features

import seaborn as sns; sns.set()
corrmat = df_full[continuous[:20]].corr()
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))
# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

# reduce the dimentions
df_full.columns[:40]
print(df_full['numsats'].isnull().sum())
print(df_full['numoprevtl'].isnull().sum())

fig=plt.figure(figsize=(15,10))
plt.subplot(1,2, 1)
plt.violinplot(df_full['numsats'],showmeans=True)
plt.title('numsats')
    
plt.subplot(1,2, 1)
plt.violinplot(df_full['numoprevtl'],showmeans=True)
plt.title('numoprevtl')

df_full.drop(['numoprevtl'], axis=1, inplace=True)
df_full.drop('mthssincelastmajorderog', inplace=True,axis=1)

import seaborn as sns; sns.set()
corrmat = df_full[continuous[21:40]].corr()
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))
# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

import seaborn as sns; sns.set()
corrmat = df_full[continuous[41:60]].corr()
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))
# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

# reduce the dimentions
print(df_full['loanamnt'].isnull().sum())
print(df_full['fundedamnt'].isnull().sum())

fig=plt.figure(figsize=(15,10))
plt.subplot(1,2, 1)
plt.violinplot(df_full['loanamnt'],showmeans=True)
plt.title('loanamnt')
    
plt.subplot(1,2, 2)
plt.violinplot(df_full['fundedamnt'],showmeans=True)
plt.title('fundedamnt')

df_full.drop(['fundedamnt'], axis=1, inplace=True)


# save the data
df_full.to_csv('C:/Users/shenbingdy/Desktop/datalab/finTech/data/df_full.csv')

