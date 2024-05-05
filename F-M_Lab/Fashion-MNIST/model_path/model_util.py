import os
import numpy as np
import matplotlib.pyplot as plt

def one_hot(n_classes,y):  #独热编码
    return np.eye(n_classes)[y]

# EPSILON=1e-8
def nll(Y_true,Y_pred):   #交叉熵损失
    Y_true=np.atleast_2d(Y_true)
    Y_pred=np.atleast_2d(Y_pred)
    loglikelihoods = np.sum(np.log(1e-8 + Y_pred) * Y_true, axis=-1)
    return -np.mean(loglikelihoods)

def softmax(X):  
    exp = np.exp(X)
    return exp/np.sum(exp,axis=-1,keepdims=True)

def sigmoid(X):  #激活函数1
    return 1 / (1 + np.exp(-X))

def relu(x):   #激活函数2
    return np.maximum(0, x)

def tanh(x):   #激活函数3
    return np.tanh(x)

def dsigmoid(X):
    sig=sigmoid(X)
    return sig * (1 - sig)


def l2_regularization(W, lambda_):   #l2_regular
    """
    计算 L2 正则化项的损失
    参数：
    - W:权重矩阵
    - lambda_:正则化强度
    返回：
    - 正则化项的损失
    """
    l2_loss = 0.5 * lambda_ * np.sum(W ** 2)
    return l2_loss

    