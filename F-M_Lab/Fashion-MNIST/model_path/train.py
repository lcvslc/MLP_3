from args import parse_arguments
import sys
sys.path.append('/Localize/lc/homework/F-M_Lab/Fashion-MNIST')
import os
import numpy as np
import matplotlib.pyplot as plt
from model_path.model import NeuralNet,LearningRateScheduler
from model_path.model_util import *
from dataprepare.get_data import getdata

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

BEST_MODEL_PATH ='model/best_model.npz'

def train():

    print("Evaluation of the untrained model:")
    n_features = train_x.shape[1]
    n_classes = len(np.unique(train_y))

    model = NeuralNet(n_features, n_hidden, n_classes)
    model.loss(train_x, train_y)
    model.accuracy(train_x, train_y)

    losses, accuracies, accuracies_test = [], [], []
    losses.append(model.loss(train_x, train_y))
    accuracies.append(model.accuracy(train_x, train_y))
    accuracies_test.append(model.accuracy(test_x, test_y))

    print("Random init: train loss: %0.5f, train acc: %0.3f, test acc: %0.3f"
        % (losses[-1], accuracies[-1], accuracies_test[-1]))

    best_acc= 0
    lr_scheduler = LearningRateScheduler(initial_lr=learning_rate, decay_steps=20, decay_rate=0.8)

    for epoch in range(10):
        # 计算测试集准确率
        test_acc = model.accuracy(test_x, test_y)
        # print(f"Epoch #{epoch + 1}, Test Accuracy: {test_acc}")
        for step in range(2):
            lr = lr_scheduler.step()
            for i, (x, y) in enumerate(zip(train_x, train_y)):
                model.train(x, y, lr)
            # print(f"Epoch {epoch + 1}, Step {step + 1}, Learning Rate: {lr}")

            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
                model.save_weights(checkpoint_file)

        # 记录训练损失和准确率
        losses.append(model.loss(train_x, train_y))
        accuracies.append(model.accuracy(train_x, train_y))
        # accuracies_test.append(test_acc)
        accuracies_test.append(model.accuracy(test_x, test_y))
        print("Epoch #%d, train loss: %0.5f, train acc: %0.3f, test acc: %0.3f"
            % (epoch + 1, losses[-1], accuracies[-1],accuracies_test[-1]))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training loss");
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='train')
    plt.plot(accuracies_test, label='test')
    plt.ylim(0, 1.1)
    plt.ylabel("accuracy")
    plt.legend(loc='best');
    plt.show()

    if sample_idx is not None:
        plot_prediction(model, sample_idx=sample_idx)


if __name__=="__main__":
    args = parse_arguments()
    print('===============================================')
    print(str(args).replace(",", "\n"), "\n")
    print('===============================================')
    n_hidden = args.n_hidden
    learning_rate = args.lr
    sample_idx=args.sample_idx
    checkpoint_file=args.checkpoint_path
    train()
