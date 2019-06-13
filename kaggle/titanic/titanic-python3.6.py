#!/usr/bin/python
# coding: utf-8
'''
Created on 2017-10-26
Update  on 2017-10-26
Author: 片刻
Github: https://github.com/apachecn/kaggle
'''

# 导入相关数据包
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

root_path = 'C:/PythonWorkSp/input/'

train_data = pd.read_csv('%s/%s' % (root_path, 'train.csv'))
dataset = train_data
# test_data = pd.read_csv('%s/%s' % (root_path, 'test.csv'))

train_data['homeNum']  = train_data.SibSp+train_data.Parch


full_len = 0
# dataset['Surname'] = dataset['Name'].apply(lambda x : x.split('.')[1])
# Surname =  dataset['Surname'].value_counts().reset_index()
 #从姓名中提取出姓
dataset['Surname'] = dataset['Name'].apply(lambda x: x.split('.')[1])
#计算数量,然后合并数据集
Surname_sum = dataset['Surname'].value_counts().reset_index()
Surname_sum.columns=['Surname','Surname_sum']
dataset = pd.merge(dataset,Surname_sum,how='left',on='Surname')
#由于出现一次时该特征时无效特征,用one来代替出现一次的姓
# dataset.loc[dataset['Surname_sum'] == 1 , 'Surname_new'] = 'one'
# dataset.loc[dataset['Surname_sum'] > 1 , 'Surname_new'] = dataset['Surname']
dataset.loc[dataset['Surname_sum'] == 1 , 'Surname_new'] = 0
dataset.loc[dataset['Surname_sum'] > 1 , 'Surname_new'] = 1
del dataset['Surname']
# dataset['Surname_sum'] = datasetTest['Surname_sum']
#分列处理
train_data['Surname_sum'] = dataset['Surname_sum']
train_data['Surname_new'] = dataset['Surname_new']


# train_data['homeNum'] = pd.concat([train_data, homeNum], axis=1)

# print(train_data.head(5))
# print(train_data.info())
# # 返回数值型变量的统计量
# print(train_data.describe())

# train_data = train_data.rename(columns={0:'homeNum'})

# print(train_data.Survived.value_counts())
train_corr = train_data.drop('PassengerId',axis=1).corr()
# print(train_corr)

a = plt.subplots(figsize=(15,9)) #调整画布大小
a = sns.heatmap(train_corr, vmin=-1, vmax=1 , annot=True , square=True)#画热力图
# plt.show()
# pclass = train_data.groupby(['Pclass'])['Pclass','Survived'].mean()
# train_data['SibSp_Parch'] = train_data['Parch']+train_data['SibSp']
# train_data[['Parch','Survived']].groupby(['Parch']).mean().plot.bar()

# print(pclass.values)

# sns.countplot('Embarked',hue='Survived',data=train_data)
# g = sns.FacetGrid(train_data, col='Survived',size=5)
# g.map(plt.hist, 'Age', bins=40)
# train_data.groupby(['Age'])['Survived'].mean().plot()
# pclass['Survived'].plot.bar()
# train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()
# plt.show()


plt.show()
