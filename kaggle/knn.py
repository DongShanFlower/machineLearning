#!/usr/bin/python
# coding: utf-8

import os
import csv
import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#data_dir = '/opt/data/kaggle/'  

# 加载数据
def opencsv():
    #pandas打开文件
    data = pd.read_csv(os.path.join('C:/PythonWorkSp/train.csv'))
    # data1 = pd.read_csv(os.path.join('C:/PythonWorkSp/test.csv'))

    train_data = data.values[:,1:]
    train_label = data.values[:,0]
    # test_data = data1.values[:,:]
    test_data = data.values[:,1:]
    return train_data, train_label, test_data

# PCA降维
def dRPCA(x_train, x_test, COMPONENT_NUM):
    pca = PCA(n_components=COMPONENT_NUM, whiten=False)
    pca.fit(x_train)
    pcaTrainData = pca.transform(x_train)
    pcaTestData = pca.transform(x_test)
    # pca 方差大小、方差占比、特征数量
    # print("方差大小:\n", pca.explained_variance_, "方差占比:\n", pca.explained_variance_ratio_)
    print("特征数量: %s" % pca.n_components_)
    print("总方差占比: %s" % sum(pca.explained_variance_ratio_))
    return pcaTrainData, pcaTestData

def trainModel(trainDataPCA,train_label):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(trainDataPCA,np.ravel(train_label))
    return clf

# 输出结果
def saveResult(result, csvName):
    with open(csvName,'w') as myFile :
        myWriter = csv.writer(myFile)
        myWriter.writerow(["ImageId","Label"])
        index = 0
        for r in result: 
            index += 1
            myWriter.writerow([index,int(r)])
    print('Saved successfully...')  # 保存预测结果

def dRecognition_knn():
    # start time
    sta_time = datetime.datetime.now()

    # load data
    train_data, train_label, test_data = opencsv()
    #test做交叉验证
    train_data, test_data, train_label, y_test = train_test_split(train_data, train_label, test_size=0.2, random_state=5)
    #降维处理
    trainDataPCA, testDataPCA = dRPCA(train_data,test_data,0.75)
    #模型训练
    clf = trainModel(trainDataPCA,train_label)
    #模型预测
    testLabel = clf.predict(testDataPCA)
    errNum = 0
    for i in range(len(testLabel)):
        if testLabel[i] != y_test[i]:
            errNum += 1
    print (errNum/len(testLabel))
    # 结果的输出
    # saveResult(testLabel,'C:/PythonWorkSp/output/Result_knn2.csv')
    print('finish')

    #结束时间
    end_time = datetime.datetime.now()
    times = (end_time - sta_time).seconds
    print("\n运行时间： %ss == %sm == %s h\n\n" % (times,times/60,times/3600))


if __name__ == "__main__":
    dRecognition_knn()