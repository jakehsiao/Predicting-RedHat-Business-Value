# -*- coding: utf-8 -*-
"""
@author: jhsiao
Main code of the MLND Capstone project
"""

import random,time,datetime
import numpy as np
import pandas as pd

# todo: read the data and print the head
atrain= pd.read_csv("act_train.csv",dtype={'people_id': np.str, 'activity_id': np.str, 'outcome': np.int8}, parse_dates=['date'])
atest  = pd.read_csv("act_test.csv", dtype={'people_id': np.str, 'activity_id': np.str}, parse_dates=['date'])
ppl    = pd.read_csv("people.csv", dtype={'people_id': np.str, 'activity_id': np.str, 'char_38': np.int32}, parse_dates=['date'])

# todo: report the shape
print 'train data:',atrain.shape
print 'test data:',atest.shape
print 'people data:',ppl.shape

# todo: set the random state
seed=2016

def treat(dataset):
    for col in list(dataset.columns):
        if col not in ['people_id', 'activity_id', 'date', 'char_38', 'outcome']:
            if dataset[col].dtype == 'object':
                dataset[col].fillna('type 0', inplace=True)
                dataset[col] = dataset[col].apply(lambda x: x.split(' ')[1]).astype(np.int32)
            elif dataset[col].dtype == 'bool':
                dataset[col] = dataset[col].astype(np.int8)
    
    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    dataset['day'] = dataset['date'].dt.day
    dataset['isweekend'] = (dataset['date'].dt.weekday >= 5).astype(int)
    
# todo: make some change to dataset to make them easier to use
treat(atrain)
treat(atest)
treat(ppl)

# set the train index for data split
train_index=2197291

# set the labels
y=atrain['outcome']

# todo: make intersection analysis on people_id
def intersection(seta,setb):
    return list(seta&setb)
intersection(set(atrain['people_id']),set(atest['people_id']))

# on group
group_set=intersection(set(train["group_1"]),set(test["group_1"]))
len(set(test["group_1"])-set(group_set))

# todo: merge the data
del atrain["outcome"]
train = atrain.merge(ppl, on='people_id', how='left')
test  = atest.merge(ppl, on='people_id', how='left')

# todo: clean the memory
del atrain
del atest
del ppl

# todo: concat the train and test data
X_all=pd.concat([train,test],ignore_index=True)

# todo: get the test activity_id s
test_aid=test["activity_id"]

# todo: drop the unuseful features
X_all=X_all.drop(["people_id","activity_id","date_x","date_y","char_10_x","group_1"],axis=1)

# todo: describe the features
features=X_all.columns
for f in features:
    print f,':',len(set(X_all[f]))
    
# todo: outlier detection
values=train["char_38"]
q1=np.percentile(values,25)
q3=np.percentile(values,75)
step=(q3-q1)*1.5
train[~((train["char_38"]>=q1-step)&(train["char_38"]<=q3+step))]

# todo: eda
from matplotlib import pyplot as plt
def simple_eda(feature):
    print feature
    pd.concat([train[feature],y],axis=1).groupby(feature).mean().hist(label=feature)
    
simple_eda_list=["char_1_x","char_2_x","char_10_x","char_3_y","char_4_y","char_7_y","group_1"]
for feature in simple_eda_list:
    simple_eda(feature)
    
# todo: make scorer
from sklearn.metrics import roc_auc_score,make_scorer
auc=make_scorer(roc_auc_score)

# todo: feature selection
features=[u'activity_category',u'char_1_y', u'char_2_y', 
        u'char_5_y', u'char_6_y', 
       u'char_8_y', u'char_9_y', u'char_10_y', u'char_11', u'char_12',
       u'char_13', u'char_14', u'char_15', u'char_16', u'char_17', u'char_18',
       u'char_19', u'char_20', u'char_21', u'char_22', u'char_23', u'char_24',
       u'char_25', u'char_26', u'char_27', u'char_28', u'char_29', u'char_30',
       u'char_31', u'char_32', u'char_33', u'char_34', u'char_35', u'char_36',
       u'char_37', u'char_38']
       
X=X_all[features].as_matrix()
# todo: transform the data as EDA steps indicated
from sklearn.preprocessing import OneHotEncoder 
hot_indexes=[0,2,3,4,5,6]
enc=OneHotEncoder(categorical_features=hot_indexes)
enc.fit_transform(X)
# todo: dimention reduction
from sklearn.decomposition import PCA
X_reduced=X
pca=PCA(n_components='mle')
pca.fit_transform(X_reduced)
# todo: train and TEST split
X_train=X_reduced[:train_index,:]
X_test=X_reduced[train_index:,:]
X_train.shape,X_test.shape
y=y.as_matrix()
y.shape

# todo: simple grid search
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression as lrc
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

def simple_gridsearch(clf,params):
    t0=time.time()
    grid=GridSearchCV(clf,params,auc,random_state=seed)
    grid.fit(X_train,y)
    print grid.best_estimator_,grid.best_score_
    print 'time used:',time.time()-t0
    return grid.best_estimator_
    
# try logistic regression
lr=lrc(random_state=seed)
params={"C":[10000,15000]}
lr1=simple_gridsearch(lr,params)

# try random forest
rf=RandomForestClassifier(random_state=seed)
params={'n_estimators':[250,500]}
rf1=simple_gridsearch(rf,params)

# try gaussian bayes
t0=time.time()
nb=gnb()
Xtrain,Xtest,ytrain,ytest=train_test_split(X_train,y)
nb.fit(Xtrain,ytrain)
prediction=nb.predict(Xtest)
print roc_auc_score(ytest,prediction)
print 'timeused: ',time.time()-t0

# try KNN
knn=KNeighborsClassifier()
params={"n_neighbors":[5,10,15,20]}
knn1=simple_gridsearch(knn,params)

# try gradient boosting
gb=GradientBoostingClassifier(random_state=seed)
params={"n_estimators":[100,250,500]}
gb1=simple_gridsearch(gb,params)

# keepTuning

LR=simple_gridsearch(lr,{"C":[8000,10000,12000]})
RF=simple_gridsearch(rf,{"n_estimators":[400,500,600]})
GB=simple_gridsearch(gb,{"n_estimators":[500,750,1000]})

# create prediction
def Pred(clfs,testdata):
    preds=np.zeros(testdata.shape[0]) #init the preds
    for clf in clf:
        preds+=(clf.predict(testdata))
    return preds/len(clfs)

# todo: 5-fold cv on final model
clfs=[LR,RF,GB]
from sklearn.cross_validation import kFold
def kFoldCV(clfs):
    kf=kFold(X_train.shape[0],5,random_state=seed)
    for train_index,test_index in kf:
        Xtrain,Xtest,ytrain,ytest=X_train[train_index,:],X_test[test_index,:],y[train_index],y[test_index]
        for clf in clfs:
            clf.fit(Xtrain,ytrain)
        print roc_auc_score(ytest,Pred(clfs,X_test))


kFoldCV(clfs)





# create prediction with leakage fills
def ensemblePred(clfs):
    sub=pd.read_csv("Submission_testsetdt.csv")
    ind=sub[sub["outcome"].isnull()].index.values
    preds=np.zeros(len(ind))
    for clf in clfs:
        preds+=(clf.predict(X_test[ind,:]))
    return preds/len(clfs)

# submission file creation
def newSub(pred):
    sub=pd.read_csv("Submission_testsetdt.csv") #leakage fills
    ind=sub[sub["outcome"].isnull()].index.values
    new=pd.concat([sub.iloc[ind,0].reset_index(drop=True),pd.Series(pred)],axis=1)
    sub2=sub.merge(new,how="left",on="activity_id")
    sub2.fillna(0,inplace=True)
    sub2["outcome"]+=sub2[0]
    sub2=sub2[["activity_id","outcome"]]
    now = datetime.datetime.now()
    sub_file = 'submission_' + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    sub2.to_csv(sub_file,index=False)
    
# todo: create submission
    
LR.fit(X_train,y)
print 'training:',time.time()-t0
RF.fit(X_train,y)
print 'training:',time.time()-t0
GB.fit(X_train,y)
print 'training:',time.time()-t0
newSub(ensemblePred([LR,RF,GB]))
print 'timeused:',time.time()-t0
print 'finished, just submit them'





