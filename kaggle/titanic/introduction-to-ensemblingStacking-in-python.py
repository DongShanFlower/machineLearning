# -*- coding: utf-8 -*-


# 加载用到的库
import pandas as pd
import numpy as np
import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# import plotly.offline as py
# py.init_notebook_mode(connected=True)
# import plotly.graph_objs as go
# import plotly.tools as tls


import warnings
warnings.filterwarnings('ignore')

# 打算使用 5 种 sklearn 中的模型来拟合
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
#, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# 加载最初的 train 和 test 数据集
train = pd.read_csv('C:/PythonWorkSp/input/train.csv')
test = pd.read_csv('C:/PythonWorkSp/input/test.csv')

# 存储 test 数据集的PassengerId 以便后面生成 submission 文件
PassengerId = test['PassengerId']

# train.head(3)

# 整体的全部数据
full_data = [train, test]

# 一些从原始 features 中衍生出的 features ，我认为可以算是一个 feature
# Name_length 特征
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
# 在 Titanic 上是否具有一个 cabin
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# 创建一个新的 feature FamilySize 作为 SibSp 和 Parch 的混合 features
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# 从 FamilySize 特征中 创建一个新的 feature IsAlone 
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# 删除 Embarked 列中的所有的 NULLS值，使用 S 来填充（算是超级暴力的吧）
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# 删除 Fare 中所有的 NULLS 值，并生成一个新的 feature 列 CategoricalFare
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    dataset['Fare'] = pd.qcut(dataset['Fare'], 4,labels=[0,1,2,3])
# CategoricalFare = pd.qcut(train['Fare'],4,labels=['0','1','2','3'])
                 
# train['CategoricalFare'] = pd.qcut(train['Fare'], 4,labels=[0,1,2,3])
# train['Fare'] = pd.qcut(train['Fare'], 4,labels=[0,1,2,3])
# 创建一个新特征 CategoricalAge



for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    dataset['Age'] = pd.cut(dataset['Age'], 5,labels=[0,1,2,3,4])

# train['CategoricalAge'] = pd.cut(train['Age'], 5,labels=[0,1,2,3,4])
# train['Age'] = pd.cut(train['Age'], 5,labels=[0,1,2,3,4])
# 定义函数从 passenger 名字中获取 titles
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # 如果 title 存在，返回
    if title_search:
        return title_search.group(1)
    return ""

# 创建一个新的特征 Title, 包含乘客的名字的 titles
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# 创建姓作为新特征:


# 将所有的非常见的 title 分类到一个 “Rare” 特征
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss') #笔误修正
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# 将数据集中的数据映射成离散型数据
for dataset in full_data:
    # 映射 Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # 映射 titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # 映射 Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # 映射 Fare
    # dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    # dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    # dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    # dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    # dataset['Fare'] = dataset['Fare'].astype(int)
    
    # 映射 Age
    # dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    # dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    # dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    # dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    # dataset.loc[ dataset['Age'] > 64, 'Age'] = 4


# train.head(3)
# full_len = 0
# for dataset in full_data:
#     # dataset['Surname'] = dataset['Name'].apply(lambda x : x.split('.')[1])
#     # Surname =  dataset['Surname'].value_counts().reset_index()
#      #从姓名中提取出姓
#     dataset['Surname'] = dataset['Name'].apply(lambda x: x.split('.')[1])
#     #计算数量,然后合并数据集
#     Surname_sum = dataset['Surname'].value_counts().reset_index()
#     Surname_sum.columns=['Surname','Surname_sum']
#     dataset = pd.merge(dataset,Surname_sum,how='left',on='Surname')
#     #由于出现一次时该特征时无效特征,用one来代替出现一次的姓
#     # dataset.loc[dataset['Surname_sum'] == 1 , 'Surname_new'] = 'one'
#     # dataset.loc[dataset['Surname_sum'] > 1 , 'Surname_new'] = dataset['Surname']
#     dataset.loc[dataset['Surname_sum'] == 1 , 'Surname_new'] = 0
#     dataset.loc[dataset['Surname_sum'] > 1 , 'Surname_new'] = 1
#     del dataset['Surname']
#     # dataset['Surname_sum'] = datasetTest['Surname_sum']
#     #分列处理
#     full_data[full_len] = pd.get_dummies(dataset,columns=['Surname_new'])
#     full_len += 1
# 生成一个临时文件，专门存储我们已经转化/映射完成之后的 train 和 test 文件，我写的很烂，，，
train = full_data[0]
test = full_data[1]
# train['Parch'] = train['Parch']+train['SibSp']
# test['Parch'] = test['Parch']+test['SibSp']
train = train.drop(['PassengerId','Name','SibSp','Ticket','Name_length','Cabin'],axis = 1)
test = test.drop(['PassengerId','Name','SibSp','Ticket','Name_length','Cabin'],axis = 1)

def saveTmpTrainFile(tmpFile,csvName):
    with open(csvName, 'w', newline='') as myFile:
        myWriter=csv.writer(myFile)
        myWriter.writerow(["Survived","Pclass","Sex","Age","Parch","Fare","Embarked","Has_cabin","FamilySize","IsAlone","Title"])
        for lines in tmpFile.index:
            tmp=[]

            tmp.append(tmpFile.loc[lines].values[1])
            tmp.append(tmpFile.loc[lines].values[2])
            tmp.append(tmpFile.loc[lines].values[4])
            tmp.append(tmpFile.loc[lines].values[5])
            tmp.append(tmpFile.loc[lines].values[7])
            tmp.append(tmpFile.loc[lines].values[9])
            tmp.append(tmpFile.loc[lines].values[11])
            # tmp.append(tmpFile.loc[lines].values[12])
            tmp.append(tmpFile.loc[lines].values[13])
            tmp.append(tmpFile.loc[lines].values[14])
            tmp.append(tmpFile.loc[lines].values[15])
            # tmp.append(tmpFile.loc[lines].values[-2])
            tmp.append(tmpFile.loc[lines].values[-1])   
            myWriter.writerow(tmp)
# train = full_data[0]            
# saveTmpTrainFile(train,'C:/PythonWorkSp/input/train_later.csv')
train.to_csv('C:/PythonWorkSp/input/train_later.csv',index=False)
# 生成一个临时文件，专门存储我们已经转化/映射完成之后的 train 和 test 文件
def saveTmpTestFile(tmpFile,csvName):
    with open(csvName, 'w', newline='') as myFile:
        myWriter=csv.writer(myFile)
        myWriter.writerow(["Pclass","Sex","Age","Parch","Fare","Embarked","Has_cabin","FamilySize","IsAlone","Title"])
        for lines in tmpFile.index:
            tmp=[]
            tmp.append(tmpFile.loc[lines].values[1])
            tmp.append(tmpFile.loc[lines].values[3])
            tmp.append(tmpFile.loc[lines].values[4])
            tmp.append(tmpFile.loc[lines].values[6])
            tmp.append(tmpFile.loc[lines].values[8])
            tmp.append(tmpFile.loc[lines].values[10])
            # tmp.append(tmpFile.loc[lines].values[11])
            tmp.append(tmpFile.loc[lines].values[12])
            tmp.append(tmpFile.loc[lines].values[13])
            tmp.append(tmpFile.loc[lines].values[14])
            # tmp.append(tmpFile.loc[lines].values[-2])
            tmp.append(tmpFile.loc[lines].values[-1]) 
            myWriter.writerow(tmp)

# def fun_test_scale(feature_scale, df_feature):
#     np_feature = df_feature.values.reshape(-1,1).astype(np.float64)
#     feature_scaled = StandardScaler().fit_transform(np_feature, feature_scale)
#     return feature_scaled

# data_test['Pclass_scaled'] = fun_test_scale(Pclass_scale, data_test['Pclass'])
# 存储 test 文件
# test = full_data[1]  
# saveTmpTestFile(test,'C:/PythonWorkSp/input/test_later.csv')
test.to_csv('C:/PythonWorkSp/input/test_later.csv',index=False)
# # ------------------------------------------------------------------------

# 加载我们生成的 train 和 test 文件
# # -----------注意一下哈，这个地方，需要调整一下生成的文件-------------------------------------------------------------
train_later = pd.read_csv('C:/PythonWorkSp/input/train_later.csv')
test_later = pd.read_csv('C:/PythonWorkSp/input/test_later.csv')

# train.head(3)
# test.head(3)

# # 使用 seaborn 将 特征的 Person 系数画出来
# colormap = plt.cm.RdBu
# plt.figure(figsize=(14,12))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
#             square=True, cmap=colormap, linecolor='white', annot=True)

# 一些比较有用参数，后边会派上用场
ntrain = train_later.shape[0]
ntest = test_later.shape[0]
SEED = 0 # 重现性
NFOLDS = 5 # 设置交叉验证的折数，以便后面的 K 折交叉验证
kf = KFold(n_splits= NFOLDS, random_state=SEED)

# sklearn 的分类器
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)

# 获取最后的 预测结果        
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    errScore = 0.0
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        y_te = y_train[test_index]
        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        y_train_p = oof_train[test_index]
        errNum = 0
        for j in range(len(y_te)):
            if y_train_p[j] != y_te[j]:
                 errNum += 1
        errScore += errNum/len(y_te)
        oof_test_skf[i, :] = clf.predict(x_test)
    print(errScore)
    oof_test[:] = oof_test_skf.mean(axis=0)
    # clf.train(x_train, y_train) 
    # scores = cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy')
    # print(scores)
    # oof_train = clf.predict(x_train)
    # oof_test = clf.predict(x_train)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# 将参数加载到调用的分类器中，进行调试
# Random Forest 的参数
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees 的参数
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost 的参数
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting 的参数
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier 的参数
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }


# 创建代表模型的 5 个对象
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
# et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
# ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
# gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
# svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# 创建 Numpy arrays 来代表 train, test 和 target
y_train = train_later['Survived'].ravel()
train_later = train_later.drop(['Survived'], axis=1)
x_train = train_later.values # 创建 train 的 array
x_test = test_later.values # 创建 test 的 array

# 创建您的 OOF 的 train 和 test 预测
# et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
# ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
# gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
# svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")

# 创建 submission 文件，接下来就是提交到 kaggle 页面看一下得分和排名啦
def saveResult(result,csvName):
    with open(csvName,'w',newline='') as myFile:    
        myWriter=csv.writer(myFile)
        myWriter.writerow(["PassengerId","Survived"])
        index=891
        for i in result:
            tmp=[]
            index=index+1
            tmp.append(index)
            tmp.append(int(i))
            myWriter.writerow(tmp)
            

# saveResult(et_oof_test,'C:/PythonWorkSp/output/result/et.csv')
saveResult(rf_oof_test,'C:/PythonWorkSp/output/result/rf.csv')
# saveResult(ada_oof_test,'C:/PythonWorkSp/output/result/ada.csv')
# saveResult(gb_oof_test,'C:/PythonWorkSp/output/result/gb.csv')
# saveResult(svc_oof_test,'C:/PythonWorkSp/output/result/svc.csv')