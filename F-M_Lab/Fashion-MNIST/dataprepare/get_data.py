import os
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn import preprocessing

# base_path=os.getcwd()
base_path='/Localize/lc/homework/F-M_Lab/Fashion-MNIST'   

transform = transforms.ToTensor()

def getdata():
    # choose the training and test datasetsy
    train_data = datasets.FashionMNIST(os.path.join(base_path,'data/'), train=True,  
                                    download=True, transform=transform)
    test_data = datasets.FashionMNIST(os.path.join(base_path,'data/'), train=False,
                                    download=True, transform=transform)

    train_xx=np.array(train_data.data,dtype=np.float32)
    train_y=np.array(train_data.targets,dtype=np.int32)
    test_xx=np.array(test_data.data,dtype=np.float32)
    test_y=np.array(test_data.targets,dtype=np.int32)

    # print('Train data size: ',train_xx.shape,train_y.shape)
    # print('Test data size: ',test_xx.shape,test_y.shape)

    scaler = preprocessing.StandardScaler()
    train_xx = train_xx.reshape((60000, -1))
    train_x = scaler.fit_transform(train_xx)
    test_xx = test_xx.reshape((10000, -1))
    test_x = scaler.transform(test_xx)

    return  train_xx, train_x, train_y,test_xx, test_x, test_y




