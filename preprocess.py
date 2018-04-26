# -*- coding: utf-8 -*-
# @author:bryan
# blog: https://blog.csdn.net/bryan__
# github: https://github.com/YouChouNoBB/sklearn-introduction/
import pandas as pd
import numpy as np
path="d:/IJCAI/round1_ijcai_18_train_20180301.txt"
df=pd.read_csv(path,sep=' ')

#数据概览
df.info()
df.describe()


#数据查看
df.head() #查看表头
df.shape() #查看行列
df.dtypes #查看数据类型
df["user_age_level"].hist() #查看变量分布
df.isnull().sum()  #查看每一列缺失值情况
df['n_null'] = df.isnull().sum(axis=1) #查看每一行缺失值情况
df["user_age_level"].value_counts() #查看这一列的值统计
df['user_age_level'].unique() #查看数据取值

for feature in df.columns:
    df.loc[df[feature]==-1,feature]=np.nan

#缺失值填充
mode_df=df.fillna(df.mode().iloc[0],inplace=True)
middf_=df.fillna(df.median())
df["user_age_level"][df.age.isnull()]="0"  #对某一列填充


#连续特征规范化处理
from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler
scaler=StandardScaler() #0均值，单位方差
scaler=MinMaxScaler(feature_range=(0, 1)) #变换到[0,1]区间（也可以是其他固定最小最大值的区间）
scaler=Normalizer(norm='l2') # 'l1', 'l2', or 'max', optional ('l2' by default)
#变换后每个样本的各维特征的平方和为1。类似地，L1 norm则是变换后每个样本的各维特征的绝对值和为1。还有max norm，则是将每个样本的各维特征除以该样本各维特征的最大值。
df['shop_review_positive_rate']=StandardScaler().fit_transform(df['shop_review_positive_rate'].values.reshape(-1,1))
df['shop_review_positive_rate']=df['shop_review_positive_rate'].rank()

#类别特征处理
#参考：https://blog.csdn.net/bryan__/article/details/79911768
data=pd.get_dummies(df,columns=['user_gender_id'],dummy_na=True)  #获得哑变量
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from scipy import sparse
df['user_gender_id']=LabelEncoder().fit_transform(df['user_gender_id'])
data=sparse.hstack((df,OneHotEncoder().fit_transform(df['user_gender_id'])))

#文本向量化
from sklearn.feature_extraction.text import CountVectorizer
df['item_property_list']=df['item_property_list'].apply(lambda x:' '.join(x.split(';')))
item_category_list = CountVectorizer().fit_transform(df['item_category_list'])
df=sparse.hstack((item_category_list,df))


#连续特征离散化
df['pv_bins']=pd.cut(df['item_pv_level'],bins=[0,5,10,15,20]).astype('str')
df['pv_bins']=LabelEncoder().fit_transform(df['pv_bins'])

#特征二值化
from sklearn.preprocessing import Binarizer
df['item_pv_level']=Binarizer(threshold=10).fit_transform(df['item_pv_level'].values.reshape(-1,1))

#评估函数
from sklearn.metrics import accuracy_score,confusion_matrix,\
    f1_score,log_loss,mean_absolute_error,mean_squared_error,\
    precision_score,roc_auc_score
#参考：https://blog.csdn.net/heyongluoyao8/article/details/49408319
#     http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
print(log_loss(y_true=df['is_trade'],y_pred=df['is_trade']))

#交叉检验
#参考：https://blog.csdn.net/xiaodongxiexie/article/details/71915259
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(df[['item_pv_level','shop_review_positive_rate']],df['is_trade'],test_size=0.1,random_state=2018)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, make_scorer
log_scorer = make_scorer(log_loss, needs_proba=True)
lr=LogisticRegression()
scores=cross_val_score(lr,df[['item_pv_level','shop_review_positive_rate']],df['is_trade'],n_jobs=-1,cv=5,scoring=log_scorer)

#参数搜索
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from lightgbm.sklearn import LGBMClassifier
from xgboost.sklearn import XGBClassifier

model = LogisticRegression()
param_grid = {'max_iter':[20,50,100], 'C': [1e-3, 1e-2, 1e-1, 1]}
grid_search = GridSearchCV(model, param_grid, n_jobs=8, verbose=1, cv=5)
grid_search.fit(train_x, train_y)
best_parameters = grid_search.best_estimator_.get_params()
for para, val in list(best_parameters.items()):
    print(para, val)
model = LogisticRegression(max_iter=best_parameters['max_iter'], C=best_parameters['C'])
model.fit(train_x, train_y)

#模型融合
#参考：https://blog.csdn.net/bryan__/article/details/51229032
