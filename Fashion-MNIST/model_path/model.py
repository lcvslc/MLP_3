import sys
sys.path.append('/Localize/lc/homework/F-M_Lab/Fashion-MNIST')
import os
import numpy as np
import matplotlib.pyplot as plt
from model_path.model_util import *


class NeuralNet():
    """只有一层隐藏的MLP"""
    def __init__(self,input_size,hidden_size,output_size):
        self.W_h=np.random.uniform(size=(input_size,hidden_size),high=0.01,low=-0.01)
        self.b_h=np.zeros(hidden_size)
        self.W_o = np.random.uniform(size=(hidden_size, output_size), high=0.01, low=-0.01)
        self.b_o = np.zeros(output_size)
        self.output_size = output_size
    def forward(self,X,keep_activations=False):
        z_h=np.dot(X,self.W_h)+self.b_h
        h = sigmoid(z_h)  #激活函数
        z_o = np.dot(h, self.W_o) + self.b_o
        y = softmax(z_o)
        if keep_activations: 
            return y, h, z_h  
        else:
            return y
        
    def loss(self, X, y):
        return nll(one_hot(self.output_size, y), self.forward(X))
    
    def grad_loss(self, x, y_true):    #反向传播设计
        y, h, z_h = self.forward(x, keep_activations=True)
        grad_z_o = y - one_hot(self.output_size, y_true)

        grad_W_o = np.outer(h, grad_z_o)
        grad_b_o = grad_z_o
        grad_h = np.dot(grad_z_o, np.transpose(self.W_o))
        grad_z_h = grad_h * dsigmoid(z_h)
        grad_W_h = np.outer(x, grad_z_h)
        grad_b_h = grad_z_h
        grads = {"W_h": grad_W_h, "b_h": grad_b_h,
                 "W_o": grad_W_o, "b_o": grad_b_o}
        return grads
    
    def train(self, x, y, learning_rate):    #SGD优化器

        grads = self.grad_loss(x, y)
        self.W_h = self.W_h - learning_rate * grads["W_h"]
        self.b_h = self.b_h - learning_rate * grads["b_h"]
        self.W_o = self.W_o - learning_rate * grads["W_o"]
        self.b_o = self.b_o - learning_rate * grads["b_o"]

    def predict(self, X):
        if len(X.shape) == 1:
            return np.argmax(self.forward(X))
        else:
            return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y):
        y_preds = np.argmax(self.forward(X), axis=1)
        return np.mean(y_preds == y)

    def save_weights(self, file_name):
        np.savez(file_name, W1=self.W_h, B1 =self.b_h,W2=self.W_o, B2 =self.b_o)

    def load_weights(self, file_name):
        data = np.load(file_name)
        self.W_h=data['W1']
        self.b_h=data['B1']
        self.W_o = data['W2']
        self.b_o = data['B2']

    def get_weights(self,file_name):
        data = np.load(file_name)
        print('W_h:',data['W1'])
        print('b_h:',data['B1'])
        print('W_o:',data['W2'])
        print('b_o:',data['B2'])
        return data['W1'],data['B1'],data['W2'],data['B2']


class LearningRateScheduler:
    def __init__(self, initial_lr, decay_steps, decay_rate):
        self.lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.global_step = 0

    def step(self):
        if self.global_step % self.decay_steps == 0 and self.global_step != 0:
            self.lr *= self.decay_rate
        self.global_step += 1
        return self.lr
