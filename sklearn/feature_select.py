# -*- coding: utf-8 -*-
# @author:bryan
# blog: https://blog.csdn.net/bryan__
# github: https://github.com/YouChouNoBB/sklearn-introduction/
import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv('d:/dataset1.csv').drop('user_id',axis=1)
data.label.replace(-1,0,inplace=True)
data=data.fillna(0)

y=data.label
x=data.drop('label',axis=1)

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)

from scipy.stats import pearsonr
columns=X_train.columns

feature_importance=[(column,pearsonr(X_train[column],y_train)[0]) for column in columns]
feature_importance.sort(key=lambda x:x[1])

import xgboost as xgb
dtrain=xgb.DMatrix(X_train,label=y_train)
dtest=xgb.DMatrix(X_test,label=y_test)

params={
    'booster':'gbtree',
    'objective':'rank:pairwise',
    'eval_metric':'auc',
    'gama':0.1,
    'min_child_weight':2,
    'max_depth':5,
    'lambda':10,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'eta':0.01,
    'tree_method':'exact',
    'seed':0,
    'nthead':7
}

watchlist=[(dtrain,'train'),(dtest,'test')]
model=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)

delete_feature=['merchant_max_distance']
X_train=X_train[[i for i in columns if i not in delete_feature]]
X_test=X_test[[i for i in columns if i not in delete_feature]]

dtrain=xgb.DMatrix(X_train,label=y_train)
dtest=xgb.DMatrix(X_test,label=y_test)
watchlist=[(dtrain,'train'),(dtest,'test')]
model=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)
'''
from minepy import MINE
m=MINE()
m.compute_score(x,x2)
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,ShuffleSplit
import numpy as np
from sklearn.metrics import roc_auc_score
rf=RandomForestClassifier(n_estimators=300,max_depth=7,min_samples_split=10,min_samples_leaf=10,
                          n_jobs=4,random_state=0)
rf.fit(X_train,y_train)
pred=rf.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,pred))
#0.867996345046

rf=RandomForestClassifier(n_estimators=300,max_depth=7,min_samples_split=10,min_samples_leaf=10,
                          n_jobs=4,random_state=0)
feature_importance=[]
for i in range(len(columns)):
    score=cross_val_score(rf,X_train.values[:,i:i+1],y_train,scoring='r2',
                          cv=ShuffleSplit(len(X_train),3,0.3))
    feature_importance.append((columns[i],round(np.mean(score),3)))
feature_importance.sort(key=lambda x:x[1])


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)
delete_feature=['merchant_max_distance']
X_train=X_train[[i for i in columns if i not in delete_feature]]
X_test=X_test[[i for i in columns if i not in delete_feature]]
rf=RandomForestClassifier(n_estimators=300,max_depth=7,min_samples_split=10,min_samples_leaf=10,
                          n_jobs=4,random_state=0)
rf.fit(X_train,y_train)
pred=rf.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,pred))


'''
xgboost 0.866091 0.866663
'''
feature_importance=model.get_fscore()
feature_importance=sorted(feature_importance.items(),key=lambda x:x[1])

delete_feature=['this_day_user_receive_same_coupon_count']
X_train=X_train[[i for i in columns if i not in delete_feature]]
X_test=X_test[[i for i in columns if i not in delete_feature]]
dtrain=xgb.DMatrix(X_train,label=y_train)
dtest=xgb.DMatrix(X_test,label=y_test)
watchlist=[(dtrain,'train'),(dtest,'test')]
model=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)


'''
l1 0.846511249837 0.846712454002
'''
from sklearn.metrics import roc_auc_score
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)
from sklearn.linear_model import LogisticRegression
import numpy as np
lr=LogisticRegression(penalty='l1',random_state=0,n_jobs=-1).fit(X_train,y_train)
pred=lr.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,pred))

feature_importance=[(i[0],i[1]) for i in zip(columns,lr.coef_[0])]
feature_importance.sort(key=lambda x:np.abs(x[1]))

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)
delete_feature=['user_mean_distance']
X_train=X_train[[i for i in columns if i not in delete_feature]]
X_test=X_test[[i for i in columns if i not in delete_feature]]

lr=LogisticRegression(penalty='l1',random_state=0,n_jobs=-1).fit(X_train,y_train)
pred=lr.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,pred))



'''
l2  0.806342999549  0.845590243511
'''

from sklearn.metrics import roc_auc_score
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)
from sklearn.linear_model import LogisticRegression
import numpy as np
lr=LogisticRegression(penalty='l2',random_state=0,n_jobs=-1).fit(X_train,y_train)
pred=lr.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,pred))

feature_importance=[(i[0],i[1]) for i in zip(columns,lr.coef_[0])]
feature_importance.sort(key=lambda x:np.abs(x[1]),reverse=True)

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)
delete_feature=['total_coupon']
X_train=X_train[[i for i in columns if i not in delete_feature]]
X_test=X_test[[i for i in columns if i not in delete_feature]]

lr=LogisticRegression(penalty='l2',random_state=0,n_jobs=-1).fit(X_train,y_train)
pred=lr.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,pred))


from sklearn.linear_model import RandomizedLasso
from sklearn.datasets import load_boston

boston=load_boston()

X=boston['data']
Y=boston['target']
names=boston['feature_names']

rlasso=RandomizedLasso(alpha=0.025).fit(X,Y)

feature_importance= sorted(zip(names,rlasso.scores_))


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)
rf=RandomForestClassifier()
rfe=RFE(rf,n_features_to_select=1,verbose=1)
rfe.fit(X_train,y_train)

feature_importance=sorted(zip(map(lambda x:round(x,4),rfe.ranking_),columns),reverse=True)

