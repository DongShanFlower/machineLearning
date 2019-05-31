import numpy as np 
import matplotlib.pyplot as plt
import math
import sys
import datetime
# sigmoid�����ͳ�ʼ������
def sigmoid(z):
    return 1/(1+np.exp(-z))
    
def init():
    # data2 = np.loadtxt('data\data2.csv',delimiter=',')
    data = np.loadtxt('data\data.csv')
    dataMatIn = data[:, 0:-1]
    classLabels = data[:, -1]
    dataMatIn = np.insert(dataMatIn, 0, 1, axis=1) 
    return dataMatIn,classLabels

# �ݶ�
def grad_descent(dataMaIn, classLabels):
    dataMatrix = np.mat(dataMaIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    weights = np.ones((n,1)) #��ʼ���ع�ϵ��
    alpha = 0.001 #ѧϰ�ʣ�����
    maxCycle = 500 #���ѭ������
    la = 0.001
    for i in range(maxCycle):
        h = sigmoid(dataMatrix*weights)
        tmp = alpha*dataMatrix.transpose()*(h-labelMat)
        weights = weights*(1-alpha*la/m) -alpha*dataMatrix.transpose()*(h-labelMat)
        err = computer_error(weights,dataMaIn,classLabels)
        plt.scatter(i,err,s=10,c='red')
    plt.xlabel('grad_num')
    plt.ylabel('J_loss')
    plt.show()
    return weights
#����ݶ��½�
def stoc_grad_ascent_one(dataMaIn, classLabels, numIter=1000):
    m, n= np.shape(dataMatIn)
    weights = np.ones(n)
    alpha = 0.001
    tmp = 0
    num = 0
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(int(m/25)):
            alpha = 4/(1+i+j)+0.01 #��֤��ε�������������Ȼ��Ӱ��
            randIndex = np.random.randint(len(dataIndex))
            h = sigmoid(sum(dataMaIn[randIndex]*weights))
            err = h - classLabels[randIndex]
            weights = weights - alpha*err*dataMaIn[randIndex]
            del(dataIndex[randIndex])
            if num == m:
                num = 0
                err = computer_error(weights,dataMaIn,classLabels)
                tmp += 1
                plt.scatter(tmp,err,s=10,c='blue')
            num += 1
    weights = weights.reshape(-1,1)
    plt.xlabel('grad_num')
    plt.ylabel('J_loss')
    plt.show()
    return weights

def computer_error(weights, dataMaIn, classLabels):
    weights = weights.reshape(-1,1)
    dataMatrix = np.mat(dataMaIn)
    labelMat = np.mat(classLabels).transpose()
    totalError = 0
    h = sigmoid(np.dot(dataMatrix,weights))
    ones = np.mat(np.ones(len(classLabels)).reshape(-1,1))
    a = ones-h
    c = np.array(a)
    tmp = []
    for x in h:
      if x == 0:
        #   tmp.append(sys.float_info.min)
          tmp.append(sys.float_info.min)
      else:
          tmp.append(math.log(x))
    # b = np.mat([math.log(x)for x in np.array(h)]) 
    b = np.mat(tmp)
    tmp = []
    for x in c:
        if x == 0:
            tmp.append(sys.float_info.min)
        else:
            tmp.append(math.log(x))
    c = np.mat(tmp)
    # c = np.mat([math.log(x) for x in np.array(a)])
    totalError = np.multiply(labelMat,b.T)+np.multiply((ones-labelMat),c.T)
    # totalError = (y_train - np.dot(x_train, beta)) ** 2
    # totalError = np.sum(totalError, axis=0)
    err = -totalError / len(dataMaIn)
    return np.sum(err)

def stoc_grad_ascent(dataMatIn, classLabels):
    m, n = np.shape(dataMatIn)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatIn[i] * weights))  #��ֵ����
        error = h - classLabels[i] 
        weights = weights - alpha * error * dataMatIn[i]
    weights = weights.reshape(-1,1)
    return weights       

#ͼ��չʾ
def plotBestFit(weights):
    dataMatIn, classLabels = init()
    n = np.shape(dataMatIn)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if classLabels[i] == 1:
            #dataMatIn�ĵ�0��ȫ��1�����Դӵڶ��п�ʼ
            xcord1.append(dataMatIn[i][1])
            ycord1.append(dataMatIn[i][2])
        else:
            xcord2.append(dataMatIn[i][1])
            ycord2.append(dataMatIn[i][2])
    
    plt.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    plt.scatter(xcord2,ycord2,s=30,c='green')

    x = np.arange(-3,3,0.1)
    y = (-weights[0,0] - weights[1,0] * x)/weights[2,0] #��Ϊȡ����X1��X2�Ĺ�ϵ��W1*X1+W2*X2+W0=0
    plt.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
if __name__ == '__main__':
    begin = datetime.datetime.now()
    dataMatIn,classLabels = init()
    # ������ݶ��½�
    res = grad_descent(dataMatIn,classLabels)
    # ��ª����ݶ��½���ÿ����һ����������ݶ��½�
    # res = stoc_grad_ascent(dataMatIn,classLabels)
    # ����ݶ��½��Ľ���
    # res = stoc_grad_ascent_one(dataMatIn,classLabels)
    end = datetime.datetime.now()
    print(res,end-begin)
    plotBestFit(res)