from args import parse_arguments
import sys
sys.path.append('/Localize/lc/homework/F-M_Lab/Fashion-MNIST')
import os
import numpy as np
import matplotlib.pyplot as plt
from model_path.model import NeuralNet
from model_path.model_util import *
from dataprepare.get_data import getdata

BEST_MODEL_PATH ='model/best_model.npz'

train_xx, train_x, train_y,test_xx, test_x, test_y = getdata()

def plot_prediction(model, sample_idx=0, classes=range(10)):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ax0.imshow(test_xx[sample_idx].reshape(28,28))   #28*28图片
    ax0.set_title("True image label: %d" % test_y[sample_idx]); #标签

    ax1.bar(classes, one_hot(len(classes), test_y[sample_idx]), label='true')
    ax1.bar(classes, model.forward(test_x[sample_idx]), label='prediction', color="red")
    ax1.set_xticks(classes)
    prediction = model.predict(test_x[sample_idx])
    ax1.set_title('Output probabilities (prediction: %d)'
                  % prediction)
    ax1.set_xlabel('Digit class')
    ax1.legend()
    plt.show()


def eval():

    n_features = train_x.shape[1]
    n_classes = len(np.unique(train_y))

    model = NeuralNet(n_features, n_hidden, n_classes)
    model.load_weights(checkpoint_file) # 读取模型的权重
    W1, b1, W2, b2 =model.get_weights(checkpoint_file)  #权重参数可视化(参数查找)
    print('TEST Accuracy:',model.accuracy(test_x, test_y))  #测试结果


    if sample_idx is not None:
        plot_prediction(model, sample_idx=sample_idx)

    # 可视化权重矩阵的直方图
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.hist(W1.flatten(), bins=50)
    plt.title('Layer 1 Weights Histogram')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 2)
    plt.hist(b1.flatten(), bins=50)
    plt.title('Layer 1 bias Histogram')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 3)
    plt.hist(W2.flatten(), bins=50)
    plt.title('Layer 2 Weights Histogram')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 4)
    plt.hist(b2.flatten(), bins=50)
    plt.title('Layer 2 bias Histogram')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')

    plt.tight_layout() 
    plt.show()


if __name__=="__main__":
    args = parse_arguments()
    print('===============================================')
    print(str(args).replace(",", "\n"), "\n")
    print('===============================================')
    n_hidden = args.n_hidden
    sample_idx=args.sample_idx
    checkpoint_file=args.checkpoint_path
    eval()
