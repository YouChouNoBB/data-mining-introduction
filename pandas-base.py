# coding=utf-8
# @author: bryan

import pandas as pd
import numpy as np
import pymysql

#缩写
# df 任意的Pandas DataFrame对象
# s 任意的Pandas Series对象，表示一列

#导入数据
filename='D:/IJCAI/file.csv'
pd.read_csv(filename,sep=' ')#从CSV文件导入数据
df=pd.read_csv(filename,sep=' ',header=None) #没有头的文件
df.columns =['f1','f2']
pd.read_table(filename)#从限定分隔符的文本文件导入数据
pd.read_excel(filename)#从Excel文件导入数据
query='select user_id,item_id from data'
db = pymysql.connect(host='host_name', port=3600, user='user_name', passwd='123456', db='db_name',charset="utf8")
pd.read_sql(query, db)#从SQL表/库导入数据
pd.read_json(json_string)#从JSON格式的字符串导入数据
pd.read_html(url)#解析URL、字符串或者HTML文件，抽取其中的tables表格
pd.read_clipboard()#从你的粘贴板获取内容，并传给read_table()

#自己构造dataframe数据
df=pd.DataFrame([[1,2,3],[4,5,6]],columns=['f1','f2','f3'])
df=pd.DataFrame({'user_id':[1,2,3],'item_id':[12,34,56]}) # 按列构造
df=pd.DataFrame([{'user_id':1,'item_id':2},{'user_id':'3'},{'item_id':4}]) #按行构造

#导出数据
df.to_csv(filename,index=False,sep=',')#导出数据到CSV文件
df.to_excel(filename) #导出数据到Excel文件
df.to_sql(table_name, db) #导出数据到SQL表
df.to_json(filename) #以Json格式导出数据到文本文件

#查看数据
df.head(n)#查看DataFrame对象的前n行
df.tail(n)#查看DataFrame对象的最后n行
df.shape()#查看行数和列数
df.info()#查看索引、数据类型和内存信息
df.describe()#查看数值型列的汇总统计
df['user_id'].value_counts(dropna=False) #查看Series对象的唯一值和计数

#数据选取
s=df['user_id']#根据列名，并以Series的形式返回列
df[['user_id', 'item_id']]#以DataFrame形式返回多列
s.iloc[0]#按位置选取数据
s.loc['index_one']#按索引选取数据
df.iloc[0,:]#返回第一行
df.iloc[0,0]#返回第一列的第一个元素
df.sample(frac=0.5)  #采样
df.sample(n=len(df))

#数据整理

pd.isnull()#检查DataFrame对象中的空值，并返回一个Boolean数组
pd.notnull()#检查DataFrame对象中的非空值，并返回一个Boolean数组
df.dropna(axis=0)#删除所有包含空值的行
df.dropna(axis=1)#删除所有包含空值的列
df.dropna(axis=1,thresh=n)#删除所有小于n个非空值的行
df.fillna(x)#用x替换DataFrame对象中所有的空值
df.fillna(df.mode().iloc[0]) #众值填充
df.fillna(df.median()) #中位数填充
df["user_age"][df.age.isnull()]="0"  #对某一列填充
s.astype(float)#将Series中的数据类型更改为float类型
df["user_age"]=df["user_age"].astype('int') #更改某列类型
s.replace(1,'one')#用‘one’代替所有等于1的值
s.replace([1,3],['one','three'])#用'one'代替1，用'three'代替3
df.columns = ['a','b','c']#重命名列名
df.rename(columns=lambda x: x + 1)#批量更改列名
df.rename(index=lambda x: x + 1)#批量重命名索引
df.rename(columns={'old_name': 'new_ name'})#选择性更改列名
df.set_index('column_one')#更改索引列
df.reset_index(drop=True) #重置索引，主要用在各种操作之后，索引会被打乱



#数据处理#Filter 、Sort 和 GroupBy
df[df[col] > 0.5]#选择col列的值大于0.5的行
df.sort_values(by='col1',ascending=True)#按照列col1排序数据，默认升序排列
df.sort_values([col1,col2], ascending=[True,False])#先按列col1升序排列，后按col2降序排列数据
df.groupby(col)#返回一个按列col进行分组的Groupby对象
df.groupby([col1,col2])#返回一个按多列进行分组的Groupby对象
df.groupby(col1)[col2].apply(np.mean)#返回按列col1进行分组后，列col2的均值
df.pivot_table(index=col1, values=[col2,col3], aggfunc=max)#创建一个按列col1进行分组，并计算col2和col3的最大值的数据透视表
df.groupby(col1).agg(np.mean)#返回按列col1分组的所有列的均值
df.groupby('user_id',as_index=False)['is_trade'].agg({'buy':'sum','click':'count','cvr':'mean'}) #生成新的df，列是user_id,buy,click,cvr

for index,row in df.iterrows():
    # index 是行号
    # row是一行
    print(index,row['user_id'])
    break
    pass

for key,df in df.groupby('user_gender_id'):
    # key 就是user_id
    # df就是分组后的dataframe
    print(key,len(df))
    pass

df['user_id'].apply(np.mean)#对DataFrame中的每一列应用函数np.mean
df.apply(np.max,axis=1)#对DataFrame中的每一行应用函数np.max
#构造分组排序特征，比如对shop分组，对组里面的item转化率分别排序
df.groupby('shop_id',as_index=False)['item_cvr'].rank(ascending=False, method='dense')

# 数据合并
df1.append(df2)#将df2中的行添加到df1的尾部
pd.concat([df1, df2],axis=1)#按列合并
pd.concat([df1,df2],axis=0)#按行合并
pd.merge(df1,df2,on='user_id',how='inner')#对df1的列和df2的列执行SQL形式的join
#差集计算
df1=pd.DataFrame({'user_id':[1,2,3,4],'item_id':[11,22,33,44]})
df2=pd.DataFrame({'user_id':[1,2]})
df2['flag']=1
df3=pd.merge(df1,df2,on='user_id',how='left')
df3=df3[df3.flag.isnull()].drop('flag',axis=1)

# 数据统计

df.mean()#返回所有列的均值
df.corr()#返回所有列与列之间的相关系数
df.item_star_level.corr(df.is_trade)
df.count()#返回每一列中的非空值的个数
df.max()#返回每一列的最大值
df.min()#返回每一列的最小值
df.median()#返回每一列的中位数
df.std()#返回每一列的标准差
df.dtypes #查看数据类型
df["user_age_level"].hist() #查看变量分布
df.isnull().sum()  #查看每一列缺失值情况
df['n_null'] = df.isnull().sum(axis=1) #查看每一行缺失值情况
df["user_age_level"].value_counts() #查看这一列的值统计
df['user_age_level'].unique() #查看数据取值