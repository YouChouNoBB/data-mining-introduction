# -*- coding: utf-8 -*-
# @Author    :yaozzhou@tencent.com
# @Time      :2018/6/11 16:45

#随机森林4个随机，数据随机，特征随机，参数随机，分裂方式随机
#参数随机可以自己去实现
import pandas as pd
import math
import numpy as np
import gc
import random
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
class DecisionTree:
    def __init__(self,params):
        self.random_state=params['random_state']
        self.method=params['method'] #gain,ratio
        self.col_sample=params['col_sample']
        self.row_sample=params['row_sample']
        self.max_depth=params['max_depth']
        self.min_size=params['min_size']

    #数据分裂，分裂出特征值对应的数据
    def dataSplit(self,data,feature,value):
        return data[data[feature]==value]

    # 数据划分,划分成左右子树
    def dataDevide(self,data, feature, value):
        left = data[data[feature] < value]
        right = data[data[feature] >= value]
        return [left, right]

    #计算标签数
    def labelCount(self,data):
        labelCount = {}
        lc=data.groupby('label', as_index=False)['label'].agg({'count': 'count'})
        for index,row in lc.iterrows():
            labelCount[row['label']]=row['count']
        return labelCount

    #信息熵
    def entropy(self,data,feature):
        values=list(data[feature].unique())          #所有取值
        p=[len(data[data[feature]==i])/len(data) for i in values]   #不同取值的数据集
        return -np.sum([i*math.log(i,2) for i in p]) # -E(p*log(p))

    #条件熵
    def conditionalEntropy(self,data,feature):
        values=list(data[feature].unique())
        C=[data[data[feature]==i] for i in values]
        size=len(data)
        return np.sum([len(i)*self.entropy(i,'label')/size for i in C])

    #信息增益
    def gain(self,data,feature):
        return self.entropy(data,'label')- self.conditionalEntropy(data,feature)

    #信息增益率
    def ratio(self,data,feature):
        return self.gain(data,feature)/self.entropy(data,feature)

    # gini指数
    def gini(self, data, feature):
        def g(data):
            try:
                values = list(data['label'].unique())
            except:
                print(data.head())
            C = [data[data['label'] == i] for i in values]
            size = len(data)
            P = [len(i) / size for i in C]
            return np.sum([i * (1 - i) for i in P])

        values = list(data[feature].unique())
        size = len(data)
        res = [(len(i[0]) * g(i[0]) / size + len(i[1]) * g(i[1]) / size, i[2]) for i in
               [self.dataDevide(data, feature, i) + [i] for i in values]]
        res.sort(key=lambda x: x[0])
        gc.collect()
        # node = self.dataDevide(data, feature, res[0][1])
        # feature,value,loss,node:[left,right]
        # loss='loss': res[0][0]
        return {'feature': feature, 'value': res[0][1], 'loss': res[0][0]}

    #计算最优特征，ID3和C45
    def chooseBestFeature(self,data,features):
        method_dict={'gain':self.gain,'ratio':self.ratio}
        feature_gain=[(method_dict[self.method](data,feature),feature) for feature in features if feature!='label']
        feature_gain.sort(key=lambda x:x[0],reverse=True)
        return feature_gain[0][1]

    #计算最优特征,CART
    def chooseBestFeatureCart(self,data):
        choose=sorted([self.gini(data,feature) for feature in data.columns if feature!='label'],key=lambda x:x['loss'])[0]
        root=self.dataDevide(data,choose['feature'],choose['value'])
        print(choose['feature'])
        return {'feature':choose['feature'],'value':choose['value'],'left':root[0],'right':root[1]}

    #随机样本选择
    def randomData(self,data):
        return data.sample(frac=self.row_sample,random_state=self.random_state)

    #随机选择特征
    def randomFeature(self,features):
        return random.sample(features,int(len(features)*self.col_sample))

    #计算叶子节点的值，取出现次数最多的值
    def getLeaf(self,data):
        return [i[0] for i in sorted(self.labelCount(data).items(), key=lambda x: x[1], reverse=True)][0]
    #构建cart
    # gini(self,data,feature)
    # feature,value,loss,node[left,right]
    def sub_split(self,root, depth):
        left, right = root['left'],root['right']
        #分裂的子树为空
        if len(left)==0 or len(right)==0:
            root['left']=root['right']=self.getLeaf(pd.concat([left,right]))
            return
        #限制深度
        if depth>self.max_depth:
            root['left']=self.getLeaf(left)
            root['right']=self.getLeaf(right)
            return
        #限制叶子节点样本数,小于阈值不再分裂,否则继续分裂
        if len(left)<self.min_size:
            root['left']=self.getLeaf(left)
        else:
            root['left']=self.chooseBestFeatureCart(left)
            self.sub_split(root['left'],depth+1)

        if len(right)<self.min_size:
            root['right']=self.getLeaf(right)
        else:
            root['right']=self.chooseBestFeatureCart(right)
            self.sub_split(root['right'],depth+1)

    #构建CART
    def createCart(self,data):
        root = self.chooseBestFeatureCart(data)
        self.sub_split(root, 1)
        return root

    #构建id3，c45
    def createTree(self,data,features):
        #所有label都相同，停止分裂
        if len(data['label'].unique())==1:
            try:
                return data['label'][0]
            except:
                return data['label'].values[0]
        #没有特征可分裂
        if len(features)==0:
            return self.getLeaf(data)
        #可分裂选择特征分裂
        bestFeature=self.chooseBestFeature(data,features)
        print(bestFeature)
        features.remove(bestFeature)
        #用字典构造决策树
        tree={}
        tree['feature']=bestFeature
        #拿到特征取值
        for value in list(data[bestFeature].unique()):
            #对分裂点的数据递归建树
            sub_data=self.dataSplit(data,bestFeature,value)
            if len(sub_data)==0:
                continue
            tree[value] = self.createTree(sub_data, features)
        return tree

    #训练
    def fit(self,X,y):
        features=self.randomFeature([i for i in X.columns if i!='label'])
        X['label']=y
        X=self.randomData(X)
        if self.method=='auto':
            self.method=random.sample(['gini','gain','ratio'],1)[0]
        print(self.method)
        # self.method == 'gain'
        if self.method=='gini':
            # print('fit ',features)
            self.tree=self.createCart(X[features+['label']])
        else:
            self.tree = self.createTree(X, features)
        return self

    def cart_predict(self,tree,row):
        if row[tree['feature']] < tree['value']:
            if isinstance(tree['left'], dict):
                return self.cart_predict(tree['left'], row)
            else:
                return tree['left']
        else:
            if isinstance(tree['right'], dict):
                return self.cart_predict(tree['right'], row)
            else:
                return tree['right']

    def id3c45_predict(self,tree,row):
        if type(tree) != dict:
            return tree
        else:
            return self.id3c45_predict(tree[row[tree['feature']]], row)
    #预测
    def predict(self,X):
        if self.method=='gini':
            return X.apply(lambda x:self.cart_predict(self.tree,x),axis=1).values
        return X.apply(lambda x:self.id3c45_predict(self.tree,x),axis=1).values

class RandomForest(DecisionTree):
    def __init__(self,params):
        # DecisionTree.__init__(params)
        self.n_estimators=params['n_estimators']

    def fit(self,X,y):
        self.trees=[]
        for i in range(self.n_estimators):
            cls=DecisionTree(params)
            cls.fit(X, y)
            self.trees.append(cls)
        # self.trees=[DecisionTree(params).fit(X,y) for i in range(self.n_estimators)]

    def predict(self,X):
        res=[tree.predict(X) for tree in self.trees]
        return pd.DataFrame(res).apply(np.mean).values


if __name__ == '__main__':
    # dataSet = [[0, 0, 0, 0, 0],  # 数据集
    #            [0, 0, 0, 1, 0],
    #            [0, 1, 0, 1, 1],
    #            [0, 1, 1, 0, 1],
    #            [0, 0, 0, 0, 0],
    #            [1, 0, 0, 0, 0],
    #            [1, 0, 0, 1, 0],
    #            [1, 1, 1, 1, 1],
    #            [1, 0, 1, 2, 1],
    #            [1, 0, 1, 2, 1],
    #            [2, 0, 1, 2, 1],
    #            [2, 0, 1, 1, 1],
    #            [2, 1, 0, 1, 1],
    #            [2, 1, 0, 2, 1],
    #            [2, 0, 0, 0, 0]]
    # data = pd.DataFrame(dataSet, columns=['年龄', '有工作', '有自己的房子', '信贷情况', 'label'])
    data=pd.read_csv('E:\\competition\\ijcai18\\data\\test.csv',sep=' ')
    data=data[['user_gender_id','user_age_level', 'user_occupation_id', 'user_star_level',
                'item_price_level', 'item_sales_level','item_collected_level', 'item_pv_level'
                ,'shop_review_num_level', 'shop_review_positive_rate','shop_star_level', 'shop_score_service', 'shop_score_delivery',
                'shop_score_description', 'is_trade'
                ]]
    data=data.rename(columns={'is_trade': 'label'})
    train,test=train_test_split(data,test_size=0.05,random_state=1024)
    params={'random_state':2018,'method':'gini','col_sample':0.8,'row_sample':0.8,'max_depth':5,'min_size':10,'n_estimators':5}
    cls=DecisionTree(params)
    # cls = RandomForest(params)
    y=train.pop('label')
    X=train
    cls.fit(X,y)
    # print(cls.trees)
    pred=cls.predict(test)
    # print(pred)
    print(log_loss(test['label'],pred))
    # print(accuracy_score(test['label'], pred))
    cls=DecisionTreeClassifier()
    cls.fit(train.drop('label', axis=1), train['label'])
    pred = cls.predict(test.drop('label', axis=1))
    print(log_loss(test['label'],pred))
    # print(accuracy_score(test['label'], pred)) 