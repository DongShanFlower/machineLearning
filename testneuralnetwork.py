from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
#����tanh�������������

def sigmoid(z):
    return 1/(1+np.exp(-z))
def relu(z):
    return (abs(z) + z) / 2
def ReLuPrime(x):
    # ReLu ����
    x[x <= 0] = 0
    x[x > 0] = 1
    return x  
  
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    # plt.show()

def plotBestFit(weights):
    x = np.arange(-3,3,0.1)
    y = (-weights[0][0] - weights[0][1] * x)/weights[0][2] #��Ϊȡ����X1��X2�Ĺ�ϵ��W1*X1+W2*X2+W0=0
    plt.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    # plt.show()

#��ʧ����
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # ���򴫲�������Ԥ��ֵ
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = X.dot(W2) + b2
    a2 = np.tanh(z2)
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # ������ʧ
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    #����ʧ�ϼ����������ѡ��
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

# �������Ϊ������ѧϰ�������ҷ���ģ��
# - nn_hdim: ���ز�Ľڵ���
# - num_passes: ͨ��ѵ���������ݶ��½��Ĵ���
# - print_loss: �����True, ��ôÿ1000�ε����ʹ�ӡһ����ʧֵ
def build_model(nn_hdim, num_passes=20000, print_loss=False):
     
    # �����ֵ��ʼ��������������Ҫѧϰ��Щ����
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
 
    # ������������Ҫ���ص�����
    model = {}
     
    # �ݶ��½�
    for i in range(0, num_passes):
 
        # ���򴫲�
        z1 = X.dot(W1) + b1
        # a1 = relu(z1)
        a1 = np.tanh(z1)
        # a1 = sigmoid(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # probs = np.tanh(z2)
        # ���򴫲�
        delta3 = probs
        # print(delta3)
        delta3[range(num_examples), y] -= 1
        # print(delta3)
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
 
        # ��������� (b1 �� b2 û��������)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
 
        # �ݶ��½����²���
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
         
        # Ϊģ�ͷ����µĲ���
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
         
        # ѡ���Եش�ӡ��ʧ
        # �����������ݳޣ���Ϊ�����õ����������ݼ����������ǲ���̫Ƶ����������
        # if print_loss and i % 1000 == 0:
        #   print ("Loss after iteration %i: %f" %(i, calculate_loss(model)))
    return model


# Ԥ�������0��1��
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # ���򴫲�
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

if __name__ == '__main__':
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    # X, y = datasets.make_circles(200, factor=0.5,noise=0.10)
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    # clf = linear_model.LogisticRegressionCV()
    # clf.fit(X, y)
    # Plot the decision boundary
    # plot_decision_boundary(lambda x: clf.predict(x),X,y)
    # plt.title("Logistic Regression")
    # a, b = clf.coef_, clf.intercept_
    # weights = np.c_[b,a]
    # plotBestFit(weights)
    num_examples = len(X) # ѵ������������
    nn_input_dim = 2 # ������ά��
    nn_output_dim = 2 # ������ά��
    # �ݶ��½��Ĳ�������ֱ���ֶ���ֵ��
    epsilon = 0.01 # �ݶ��½���ѧϰ��
    reg_lambda = 0.01 # ���򻯵�ǿ��

    # �һ��3ά���ز��ģ��
    model = build_model(3, print_loss=True)
    
    # �������߽߱�
    plot_decision_boundary(lambda x: predict(model, x),X,y)
    plt.title("Decision Boundary for hidden layer size 3")

    plt.show()

