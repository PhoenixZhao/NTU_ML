#coding=utf8
'''
    Implement the fixed learning rate gradient descent algorithm for logistic regression.
    Run the algorithm with η=0.001 and T=2000
    The algorithm is slide 23/25 in lecture 10 of ML01
'''
import numpy as np
import math

train_data_file = '../../../台大ML01/assignments/03/hw3_train.dat'
test_data_file = '../../../台大ML01/assignments/03/hw3_test.dat'

eita = 0.001
T = 2000


def load_data(data_file):
    '''
        读入数据，数据格式空格分开，最后一个为label
        构造的矩阵包括:
            X: N * (d+1)矩阵,第一列为全1，后面即为特征
            Y*: 对角矩阵，-y1, ..., -yn
    '''
    d = np.loadtxt(data_file)
    r, c = d.shape
    feature_indexs = range(c-1)
    X = d[:,feature_indexs]
    Y = d[:, c-1]
    ones = np.ones((r, 1), dtype=float)
    X = np.hstack((ones, X))
    return X, Y

def logit(x):
    '''
        e^x / (1 + e^x)
    '''
    return 1.0 / (1 + pow(math.e, -x))

def sign(x):
    return 1 if x>= 0 else -1

def train_with_gd():
    X, Y = load_data(train_data_file)
    N, dn = X.shape
    W = np.zeros((dn, 1))
    Y_star = -1 * np.diag(Y)
    #gradient descent for training
    for t in range(T):
        if t % 100 == 0:
            print 'round ', t + 1
        Theta_Matrix = np.dot(np.dot(W.T, X.T), Y_star)
        #将Theta矩阵中每个元素进行logit运算
        f = np.vectorize(logit)
        Theta_Matrix = f(Theta_Matrix)
        gradient = 1.0 / N * np.dot(np.dot(Theta_Matrix, Y_star), X)
        W = W - (eita * gradient).T
    return W

def train_with_sgd():
    X, Y = load_data(train_data_file)
    N, dn = X.shape
    W = np.zeros((dn, 1))
    Y_star = -1 * Y
    #stochastic gradient descent for training
    rn = np.random.randint(N, size=T)
    for t in range(T):
        if t % 100 == 0:
            print 'round ', t + 1
        xn = X[rn[t],:].reshape(1, dn)
        theta = logit(np.dot(xn, W)[0,0])
        #将Theta矩阵中每个元素进行logit运算
        gradient =  Y_star[rn[t]] * theta * xn
        W = W - (eita * gradient).T
    return W

if __name__ == '__main__':
    W = train_with_sgd()
    #test
    X, Y = load_data(test_data_file)
    N, dn = X.shape
    pred_Y = np.dot(X, W)#得到列向量
    f = np.vectorize(sign)
    pred_Y = f(pred_Y)
    #print X.shape
    #print pred_Y.shape
    #print Y.reshape(N, 1).shape
    if eita == 0.001:
        print '********************Q18: eita=0.001, T=2000********************'
        print '********************Eout on test data**************************'
        print "Eout=", 1 - np.sum(pred_Y == Y.reshape(N, 1)) * 1.0 / N
        print '***************************************************************'

    if eita == 0.01:
        print '********************Q19: eita=0.01, T=2000********************'
        print '********************Eout on test data**************************'
        print "Eout=", 1 - np.sum(pred_Y == Y.reshape(N, 1)) * 1.0 / N
        print '***************************************************************'



