
## Import the necessary packages
import time
from sqlalchemy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-bright')
#plt.style.use('ggplot')

#load the data
df_full=pd.read_csv('C:/Users/shenbingdy/Desktop/datalab/finTech/data/df_full.csv',encoding = "ISO-8859-1")
df_full.head(5)
df_full.drop(['Unnamed: 0'],axis=1,inplace=True)
df_full.drop('addrstate', inplace=True,axis=1)
df_full.drop('zipcode', inplace=True,axis=1)
df_full.drop('emptitle', inplace=True,axis=1)
df_full.reset_index(inplace=True)
df_full.drop('index', inplace=True,axis=1)

#feature engeneering
df_full.columns

Object=[]
Number=[]
for i in df_full.columns:
    if df_full[i].dtype=='object':
        Object+=[i]
    else:
        Number+=[i]   

Object=['homeownership',
'purpose',
'initialliststatus',
'verificationstatus']

#Label Encoding and Hot Label Encoding
from sklearn import metrics, preprocessing, cross_validation
LBL = preprocessing.LabelEncoder()
start=time.time()
LE_map=dict()
for i in Object:
    df_full[i] = LBL.fit_transform( df_full[i])
    LE_map[i]=dict(zip(LBL.classes_, LBL.transform(LBL.classes_)))
print ('Label enconding finished in %f seconds' % (time.time()-start))

print(LE_map)

OHE = preprocessing.OneHotEncoder(sparse=False)
start=time.time()
OHE.fit(df_full[Object])
OHE_data=OHE.transform(df_full[Object])                              
print ('One-hot-encoding finished in %f seconds' % (time.time()-start))

OHE_vars = [var[:-3] + '_' + str(level).replace(' ','_')                for var in Object for level in LE_map[var]]
print ("OHE size :" ,OHE_data.shape)
print ("One-hot encoded catgorical feature samples : %s" % (OHE_vars[:100]))

df_OHE=pd.DataFrame(OHE_data,columns=OHE_vars)
df_full_data = pd.concat([df_full, df_OHE],axis=1)

# get all data down
# separate the train and test

df_Test=df_full_data.query('issued=="now"')
df_Train=df_full_data.query('issued!="now"')
df_Train['issued']=df_Train['issued'].astype(int)

df_train=df_Train.query('issued<=9')
df_valid=df_Train.query('issued>9')

df_valid_X=df_valid.drop(['id', 'loanstatus','issued'],axis=1)
df_valid_Y=df_valid['loanstatus']

df_train_Y=df_train['loanstatus'].astype(int)
df_train_X=df_train.drop(['id', 'loanstatus','issued'],axis=1)

df_Test_X=df_Test.drop(['id', 'loanstatus','issued'],axis=1)

# model building
import xgboost as xgb
from sklearn.cross_validation import train_test_split

train_x,test_x,train_y,test_y= train_test_split(df_train_X,df_train_Y, test_size=0.3, random_state=1000 , stratify=df_train_Y )
train_M=xgb.DMatrix(train_x,train_y,missing=np.nan)
test_M=xgb.DMatrix(test_x,test_y,missing=np.nan)
test_r_M=xgb.DMatrix(df_valid_X,df_valid_Y,missing=np.nan)

#training the data simplely

params={ #general 
        'booster':'gbtree',
        'silent':0,
         #Parameters for Tree Booster
        'etc':0.02,
        'gamma':1,
        'max-depth':10,
        'min_child-weight':5,
        'max_delta_step':0,
        'subsample':0.632,
        'colsample_bytree':0.7,
        'colsample_bylevel':1,
        'lambda':1,
        #Learning Task Parameters
        'objective':'binary:logistic',
        'seed':1234,
        'eval-metric': 'ave',
        }

watchlist=[(train_M, 'train'), (test_M, 'eval')]
model_G=xgb.train(params, dtrain=train_M, num_boost_round=1500, evals=watchlist, obj=None,              feval=None, maximize=False, early_stopping_rounds=50, evals_result=None,               verbose_eval=True, xgb_model=None, callbacks=None, learning_rates=None)


# check the results
from sklearn.metrics import roc_curve,auc
from sklearn import linear_model,datasets

def draw_roc (model, x_train, x_test, x_r_test, y_train, y_test, y_r_test):
    probas= model.predict(x_train, ntree_limit=model.best_ntree_limit)
    probas_1= model.predict(x_test, ntree_limit=model.best_ntree_limit)
    probas_2= model.predict(x_r_test, ntree_limit=model.best_ntree_limit)
    fpr,tpr,threshhold=roc_curve(y_train, probas)
    fpr_1,tpr_1,threshhold_1=roc_curve(y_test, probas_1)
    fpr_2,tpr_2,threshhold_2=roc_curve(y_r_test, probas_2)
    roc_auc=auc(fpr, tpr)
    roc_auc_1=auc(fpr_1, tpr_1)
    roc_auc_2=auc(fpr_2, tpr_2)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, label=' ROC Curve-train (AUC=%0.2f)'% roc_auc, color='r')
    plt.plot(fpr_1, tpr_1, label=' ROC Curve-test (AUC=%0.2f)'% roc_auc_1, color='b')
    plt.plot(fpr_2, tpr_2, label=' ROC Curve-valid (AUC=%0.2f)'% roc_auc_2, color='g')
    plt.plot([0,1],[0,1], 'k--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for the model')
    plt.legend(loc='low right')
    plt.show()
      
draw_roc (model_G, train_M, test_M, test_r_M, train_y, test_y, df_valid_Y)


# Tune the model 
Train=xgb.DMatrix(df_train_X,df_train_Y)

from bayes_opt import BayesianOptimization
def xgb_evaluate (min_child_weight, colsample_bytree, max_depth, subsample, gamma):
    param={'objective':'binary:logistic',
           'eta':0.1,
           'max_depth':int(max_depth),
           'min_child_weight': int(min_child_weight),
           'colsample_bytree':colsample_bytree,
           'subsample':subsample,
           'gamma':gamma,
           'verbose_eval':False}     
    #xgb.cv: Cross-validation with given parameters
    cv_results=xgb.cv(param, Train,
                     num_boost_round=1000000,
                     nfold=3,
                     metrics={'auc'},#'auc'
                     early_stopping_rounds=1000000,
                     seed=1234,
                     callbacks=[xgb.callback.early_stop(50)])
    print(cv_results)
    return cv_results['test-auc-mean'].max()
    
xgb_BO=BayesianOptimization(xgb_evaluate,
                            {'max_depth':(1,50),
                             'min_child_weight':(0,20),
                             'colsample_bytree':(0.1,1),
                             'subsample':(0.1,1),
                             'gamma':(0,2)})
xgb_BO.maximize(init_points=10, n_iter=20, acq='ucb')

print(xgb_BO.res['max'])
max_params=xgb_BO.res['max']['max_params']
max_params={'max_depth': 2, 'min_child_weight': 19, 
            'colsample_bytree': 0.96, 'subsample': 0.98352250641954819, 'gamma': 1.8161820125906867}
model_G_max=xgb.train(max_params, dtrain=train_M, num_boost_round=1500, evals=watchlist, obj=None,              feval=None, maximize=False, early_stopping_rounds=50, evals_result=None,               verbose_eval=True, xgb_model=None, callbacks=None, learning_rates=None)

# check the results: 
draw_roc (model_G_max, train_M, test_M, test_r_M, train_y, test_y, df_valid_Y)
# The valid and tedt data are much improved

df_Test_X_M=xgb.DMatrix(df_Test_X,missing=np.nan)
df_Test_Y=model_G_max.predict(df_Test_X_M)

df_Test.reset_index(inplace=True)
df_Test.drop(['index'],inplace=True,axis=1)

#get the results
df_re=pd.DataFrame(df_Test['id'])
df_re['probility']=df_Test_Y

# get the important features
importance=model_G_max.get_score(importance_type='weight')
df_im=pd.DataFrame(list(importance.items()), columns=['fea','score'])
df_im.sort_values(['score'],ascending=False,inplace=True)
#plot the data

plt.style.use('seaborn-poster')
plt.style.use('seaborn-deep')
#plt.style.use('seaborn-ticks')
plt.style.use('fivethirtyeight')
fig=plt.figure(figsize=(6,8))
plt.barh(range(10), df_im.score[:10])
plt.yticks(range(10),  df_im.fea[:10])
plt.ylabel('Feature')
plt.xlabel('Importance')
plt.show()

