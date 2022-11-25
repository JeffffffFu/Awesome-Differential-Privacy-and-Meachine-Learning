import gzip
from pathlib import Path
import pickle
import numpy
import os
import numpy as np

def load_fashionMnist(path, kind):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)

    #如果是.gz文件，则要gzip.open。如果本身就是ubyte则不需要
    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)



    return images, labels

#这个函数调用上面的函数，因为数据包格式的问题，多了上面的步骤
def dataload_fashionMnist():
    x_test, y_test = load_fashionMnist('./data/FashionMnist/', kind='t10k')
    x_train, y_train = load_fashionMnist('./data/FashionMnist/', kind='train')

    x_train = x_train.reshape(60000, 1, 28, 28)
    x_test = x_test.reshape(10000, 1, 28, 28)
    x_valid = x_test
    y_valid = y_test

    # print("x_train_shape:",x_train.shape)
    # print("y_train_shape:", y_train.shape)
    # print("x_test_shape:",x_test.shape)
    # print("y_test_shape:", y_test.shape)

    return x_train, y_train, x_valid, y_valid, x_test, y_test