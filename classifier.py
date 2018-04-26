# -*- coding: utf-8 -*-
# @author:bryan
# blog: https://blog.csdn.net/bryan__
# github: https://github.com/YouChouNoBB/sklearn-introduction/

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

data=pd.read_csv('d:/all_window.csv').fillna(0,axis=1)

x=data.drop('label',axis=1)
y=data['label']

X_train=x[:len(x)*0.9]
X_test=x[len(x)*0.9:]

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)

reg=linear_model.LinearRegression()
reg.fit(X_train,y_train)

y_pred=reg.predict(X_test)

metrics.mean_squared_error(y_test,y_pred)

reg=linear_model.Ridge(alpha=0.5)
reg=linear_model.Lasso(alpha=2,max_iter=10)


data2=pd.read_csv('d:/trainCG.csv').fillna(0,axis=1)
x=data2.drop('label',axis=1)
y=data2['label']

X_train_2,X_test_2,y_train_2,y_test_2=train_test_split(x,y,test_size=0.1,random_state=0)

cls=linear_model.LogisticRegression()
cls.fit(X_train_2,y_train_2)

y_pred=cls.predict_proba(X_test_2)
metrics.roc_auc_score(y_test_2,y_pred)


from sklearn.svm import SVR,SVC

reg=SVR()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)

metrics.mean_squared_error(y_test,y_pred)

cls=SVC(probability=True,kernel='rbf',C=0.1,max_iter=10)
cls.fit(X_train_2,y_train_2)
y_pred=cls.predict_proba(X_test_2)[:,1]
metrics.roc_auc_score(y_test_2,y_pred)

from sklearn.neural_network import MLPClassifier,MLPRegressor
reg=MLPRegressor(hidden_layer_sizes=(10,10,10),learning_rate=0.1)


from sklearn.tree import DecisionTreeClassifier
cls=DecisionTreeClassifier(max_depth=6,min_samples_split=10,min_samples_leaf=5,max_features=0.7)
cls.fit(X_train_2,y_train_2)
y_pred=cls.predict_proba(X_test_2)[:,1]
metrics.roc_auc_score(y_test_2,y_pred)


from sklearn.ensemble import RandomForestClassifier
cls=RandomForestClassifier(max_depth=6,min_samples_split=10,min_samples_leaf=5,max_features=0.7)
cls.fit(X_train_2,y_train_2)
y_pred=cls.predict_proba(X_test_2)[:,1]
metrics.roc_auc_score(y_test_2,y_pred)

from sklearn.ensemble import ExtraTreesClassifier
cls=ExtraTreesClassifier(max_depth=6,min_samples_split=10,min_samples_leaf=5,max_features=0.7)
cls.fit(X_train_2,y_train_2)
y_pred=cls.predict_proba(X_test_2)[:,1]
metrics.roc_auc_score(y_test_2,y_pred)

from sklearn.ensemble import GradientBoostingClassifier
cls=GradientBoostingClassifier(max_depth=6,min_samples_split=10,min_samples_leaf=5,max_features=0.7)
cls.fit(X_train_2,y_train_2)
y_pred=cls.predict_proba(X_test_2)[:,1]
metrics.roc_auc_score(y_test_2,y_pred)

from xgboost.sklearn import XGBClassifier
cls=XGBClassifier(max_depth=6,min_samples_split=10,min_samples_leaf=5,max_features=0.7)
cls.fit(X_train_2,y_train_2)
y_pred=cls.predict_proba(X_test_2)[:,1]
metrics.roc_auc_score(y_test_2,y_pred)

from lightgbm import LGBMClassifier,LGBMRegressor
cls=LGBMClassifier(max_depth=6,min_samples_split=10,min_samples_leaf=5,max_features=0.7)
cls.fit(X_train_2,y_train_2)
y_pred=cls.predict_proba(X_test_2)[:,1]
metrics.roc_auc_score(y_test_2,y_pred)

reg=LGBMRegressor(random_state=0,num_leaves=40,max_depth=7,n_estimators=200,subsample=0.7,colsample_bytree=0.7,reg_lambda=0.5)
reg.fit(X_train,y_train)
y_pred_lgb=reg.predict(X_test)
metrics.mean_squared_error(y_test,y_pred_lgb)
